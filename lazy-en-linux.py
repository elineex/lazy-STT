#!/usr/bin/env python3

import os
import sys
import tempfile
import time
import wave
import threading

import keyboard
import numpy as np
import pyautogui
import sounddevice as sd
from faster_whisper import WhisperModel

left_hold_active = False
left_click_loop_active = False

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.1
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION)

START_THRESHOLD = 0.015
CONTINUE_THRESHOLD = 0.010
SILENCE_HOLD_SECONDS = 1.0
MIN_SPEECH_SECONDS = 0.35
MAX_RECORD_SECONDS = 20
IDLE_SLEEP_SECONDS = 0.05

MODEL_NAME = "medium"
DEVICE = "cpu"          # "cpu" or "cuda"
COMPUTE_TYPE = "int8"   # int8 for CPU, float16 for CUDA usually

FORCE_LANGUAGE = "en"   # None = auto, "bg", "en"

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


def hold_left_mouse_until_escape() -> None:
    global left_hold_active

    if left_hold_active:
        print("[INFO] Left mouse is already being held.")
        return

    left_hold_active = True
    print("[MOUSE] Holding LEFT button. Press ESC to release.")

    try:
        pyautogui.mouseDown(button="left")
        keyboard.wait("esc")
    finally:
        pyautogui.mouseUp(button="left")
        left_hold_active = False
        print("[MOUSE] LEFT button released.")


def click_left_loop_until_escape(interval_seconds: float = 0.08) -> None:
    global left_click_loop_active

    if left_click_loop_active:
        print("[INFO] Left click loop is already running.")
        return

    left_click_loop_active = True
    print("[MOUSE] Left click loop started. Press ESC to stop.")

    try:
        while True:
            if keyboard.is_pressed("esc"):
                break
            pyautogui.mouseDown(button="left")
            time.sleep(0.01)
            pyautogui.mouseUp(button="left")
            time.sleep(interval_seconds)
    finally:
        left_click_loop_active = False
        print("[MOUSE] Left click loop stopped.")


def type_text(text: str, add_trailing_space: bool = True) -> None:
    text = text.strip()
    if not text:
        return

    if add_trailing_space:
        text += " "

    pyautogui.write(text, interval=0)


def send_special_key(name: str) -> None:
    key_map = {
        "enter": "enter",
        "tab": "tab",
        "backspace": "backspace",
        "delete": "delete",
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "page down": "pagedown",
        "page up": "pageup",
        "escape": "esc",
        "f1": "f1",
        "f2": "f2",
        "f3": "f3",
        "f4": "f4",
        "f5": "f5",
        "f6": "f6",
        "f7": "f7",
        "f8": "f8",
        "f9": "f9",
        "f10": "f10",
        "f11": "f11",
        "f12": "f12",
    }

    key = key_map.get(name)
    if key:
        pyautogui.press(key)


def send_hotkey(*keys: str) -> None:
    pyautogui.hotkey(*keys)


def audio_level(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))


def save_temp_wav(audio: np.ndarray) -> str:
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    return temp_path


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.rstrip(" ,.!?;:")
    return " ".join(text.split())


def extract_go_remainder(raw_text: str):
    cleaned = raw_text.strip().rstrip(" ,.!?;:")
    words = cleaned.split()
    if not words:
        return None

    first = words[0].lower()
    if first in {"go", "write"}:
        return " ".join(words[1:]).strip()

    return None


def wait_for_speech_and_record():
    frames = []
    speech_started = False
    speech_duration = 0.0
    silence_after_speech = 0.0
    started_at = None

    print("[WAIT] Waiting for speech...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCKSIZE,
    ) as stream:
        while True:
            data, _ = stream.read(BLOCKSIZE)
            chunk = np.squeeze(data.copy())
            level = audio_level(chunk)

            if not speech_started:
                if level >= START_THRESHOLD:
                    speech_started = True
                    started_at = time.time()
                    frames.append(chunk)
                    speech_duration += BLOCK_DURATION
                    print("[ON] Speech detected. Listening...")
                else:
                    time.sleep(IDLE_SLEEP_SECONDS)
                continue

            frames.append(chunk)

            if level >= CONTINUE_THRESHOLD:
                speech_duration += BLOCK_DURATION
                silence_after_speech = 0.0
            else:
                silence_after_speech += BLOCK_DURATION

            if started_at is not None and (time.time() - started_at) >= MAX_RECORD_SECONDS:
                print("[INFO] Maximum phrase length reached.")
                break

            if speech_duration >= MIN_SPEECH_SECONDS and silence_after_speech >= SILENCE_HOLD_SECONDS:
                break

    if not frames:
        return None

    audio = np.concatenate(frames)

    if len(audio) < int(SAMPLE_RATE * 0.25):
        return None

    return save_temp_wav(audio)


def transcribe_file(model: WhisperModel, wav_path: str):
    segments, info = model.transcribe(
        wav_path,
        language=FORCE_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    text = "".join(segment.text for segment in segments).strip()
    detected_language = getattr(info, "language", None)
    return text, detected_language


def handle_command_or_text(text: str) -> bool:
    raw = text.strip()
    normalized = normalize_text(raw)

    if not raw:
        return False

    stop_commands = {
        "stop",
        "stop listening",
        "stop program",
        "exit",
        "exit program",
        "quit",
        "quit program",
        "end",
    }

    if normalized in stop_commands:
        print("[STOP] Exit command received.")
        return True

    if normalized in {"hold left", "left hold"}:
        threading.Thread(target=hold_left_mouse_until_escape, daemon=True).start()
        return False

    if normalized in {"keep left", "click left", "left click loop", "spam left", "click left loop"}:
        threading.Thread(target=click_left_loop_until_escape, daemon=True).start()
        return False

    command_map = {
        "enter": ("key", "enter"),
        "new line": ("key", "enter"),
        "newline": ("key", "enter"),
        "tab": ("key", "tab"),
        "backspace": ("key", "backspace"),
        "delete": ("key", "delete"),
        "comma": ("text", ","),
        "period": ("text", "."),
        "dot": ("text", "."),
        "question mark": ("text", "?"),
        "exclamation mark": ("text", "!"),
        "colon": ("text", ":"),
        "semicolon": ("text", ";"),
        "up": ("key", "up"),
        "down": ("key", "down"),
        "left": ("key", "left"),
        "right": ("key", "right"),
        "rignt": ("key", "right"),
        "page down": ("key", "page down"),
        "page up": ("key", "page up"),
        "colum": ("text", "|"),
        "column": ("text", "|"),
    }

    shortcut_map = {
        "select all": ("ctrl", "a"),
        "copy": ("ctrl", "c"),
        "paste": ("ctrl", "v"),
        "past": ("ctrl", "v"),
        "f1": ("f1",),
        "f2": ("f2",),
        "f3": ("f3",),
        "f4": ("f4",),
        "f5": ("f5",),
        "f6": ("f6",),
        "f7": ("f7",),
        "f8": ("f8",),
        "f9": ("f9",),
        "f10": ("f10",),
        "f11": ("f11",),
        "f12": ("f12",),
        "escape": ("esc",),
        "alt tab": ("alt", "tab"),
        "alt+tab": ("alt", "tab"),
        "switch window": ("alt", "tab"),
        "close window": ("alt", "f4"),
        "close windows": ("alt", "f4"),
        "control v": ("ctrl", "v"),
        "ctrl a": ("ctrl", "a"),
        "control a": ("ctrl", "a"),
    }

    go_part = extract_go_remainder(raw)
    if go_part is not None:
        if go_part:
            print(f"[TEXT] Typing: {go_part}")
            type_text(go_part, add_trailing_space=True)
        else:
            print("[INFO] 'go' was spoken without text.")
        return False

    if normalized in shortcut_map:
        print(f"[SHORTCUT] Executing: {normalized}")
        send_hotkey(*shortcut_map[normalized])
        return False

    if normalized in command_map:
        kind, value = command_map[normalized]
        print(f"[CMD] Executing: {normalized}")

        if kind == "key":
            send_special_key(value)
        else:
            type_text(value, add_trailing_space=False)

        return False

    print("[IGNORE] This is not a command. For typing, use: go <text>")
    return False


def process_transcribed_text(text: str) -> bool:
    return handle_command_or_text(text)


def main() -> None:
    print("Loading Whisper model...")
    print(f"Model: {MODEL_NAME}, device={DEVICE}, compute_type={COMPUTE_TYPE}")
    print("On first run, the model may be downloaded automatically.\n")

    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    try:
        print("Available audio devices:")
        print(sd.query_devices())
        print()
    except Exception as e:
        print(f"Could not read audio devices: {e}\n")

    print("Ready.")
    print("Commands are executed directly, without a wake word.")
    print("For typing, use: go <text>")
    print("Examples:")
    print("  select all")
    print("  up")
    print("  page down")
    print("  column")
    print("  go hello world")
    print("  write this is text")
    print("To exit: Ctrl+C or say 'stop' / 'end'\n")

    try:
        while True:
            wav_path = wait_for_speech_and_record()

            if wav_path is None:
                print("[OFF] No valid phrase detected.\n")
                continue

            try:
                text, detected_language = transcribe_file(model, wav_path)
            finally:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

            if not text:
                print("[OFF] Nothing was recognized.\n")
                continue

            if detected_language:
                print(f"[LANG={detected_language}] {text}")
            else:
                print(f"[TEXT] {text}")

            should_stop = process_transcribed_text(text)
            print("[OFF] Done. Waiting for speech again.\n")

            if should_stop:
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
