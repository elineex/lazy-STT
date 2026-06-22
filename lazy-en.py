# MIT License
#
# Copyright (c) [2026] [Elin Elinov Hristov]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import subprocess
import sys
import tempfile
import time
import wave
import threading
import keyboard
import win32api
import win32con

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

left_hold_active = False
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

MODEL_NAME = "medium"      # small / medium
DEVICE = "cpu"             # cpu / cuda
COMPUTE_TYPE = "int8"      # int8 for CPU, float16 for CUDA usually

# None = auto detect, "bg" = Bulgarian, "en" = English
FORCE_LANGUAGE = "en"

def hold_left_mouse_until_escape() -> None:
    global left_hold_active

    if left_hold_active:
        print("[INFO] Left mouse is already being held.")
        return

    left_hold_active = True
    print("[MOUSE] Holding LEFT button. Press ESC to release.")

    try:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        keyboard.wait("esc")
    finally:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        left_hold_active = False
        print("[MOUSE] LEFT button released.")

left_click_loop_active = False


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
#            subprocess.run(
#              ["powershell.exe", "-NoProfile", "-STA", "-Command", "{LEFT}"],
#              check=False,
#              stdout=subprocess.DEVNULL,
#              stderr=subprocess.DEVNULL,
#            )

            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.01)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(interval_seconds)

    finally:
        left_click_loop_active = False
        print("Left click loop stopped.")

def escape_sendkeys(text: str) -> str:
    replacements = {
        "{": "{{}",
        "}": "{}}",
        "+": "{+}",
        "^": "{^}",
        "%": "{%}",
        "~": "{~}",
        "(": "{(}",
        ")": "{)}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def send_keys_raw(text: str) -> None:
    safe_text = escape_sendkeys(text)
    safe_text = safe_text.replace("'", "''")

    ps_command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        f"[System.Windows.Forms.SendKeys]::SendWait('{safe_text}')"
    )

    subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", ps_command],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def send_special_key(key_token: str) -> None:
    ps_command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        f"[System.Windows.Forms.SendKeys]::SendWait('{key_token}')"
    )

    subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", ps_command],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def send_key_sequence(sequence: str) -> None:
    ps_command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        f"[System.Windows.Forms.SendKeys]::SendWait('{sequence}')"
    )

    subprocess.run(
        ["powershell.exe", "-NoProfile", "-STA", "-Command", ps_command],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def type_text(text: str, add_trailing_space: bool = True) -> None:
    text = text.strip()
    if not text:
        return

    if add_trailing_space:
        send_keys_raw(text + " ")
    else:
        send_keys_raw(text)


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
       hold_thread = threading.Thread(target=hold_left_mouse_until_escape, daemon=True)
       hold_thread.start()
       return False

    if normalized in {"keep left", "click left", "left click loop", "spam left", "click left loop"}:
      threading.Thread(target=click_left_loop_until_escape, daemon=True).start()
      return False

    command_map = {
        "enter": "{ENTER}",
        "new line": "{ENTER}",
        "newline": "{ENTER}",
        "tab": "{TAB}",
        "backspace": "{BACKSPACE}",
        "delete": "{DELETE}",
        "comma": ",",
        "period": ".",
        "dot": ".",
        "question mark": "?",
        "exclamation mark": "!",
        "colon": ":",
        "semicolon": ";",
        "up": "{UP}",
        "down": "{DOWN}",
        "left": "{LEFT}",
        "right": "{RIGHT}",
        "rignt": "{RIGHT}",
        "page down": "{PGDN}",
        "page up": "{PGUP}",
        "colum": "|",
        "column": "|",
    }

    shortcut_map = {
        "select all": "^a",
        "copy": "^c",
        "paste": "^v",
        "past": "^v",
        "f1": "{F1}",
        "f2": "{F2}",
        "f3": "{F3}",
        "f4": "{F4}",
        "f5": "{F5}",
        "f6": "{F6}",
        "f7": "{F7}",
        "f8": "{F8}",
        "f9": "{F9}",
        "f10": "{F10}",
        "f11": "{F11}",
        "f12": "{F12}",
        "escape": "{ESC}",
        "alt tab": "%{TAB}",
        "alt+tab": "%{TAB}",
        "switch window": "%{TAB}",
        "close window": "%{F4}",
        "close windows": "%{F4}",
        "control v": "^v",
        "ctrl a": "^a",
        "control a": "^a",
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
        send_key_sequence(shortcut_map[normalized])
        return False

    if normalized in command_map:
        value = command_map[normalized]
        print(f"[CMD] Executing: {normalized}")

        if value.startswith("{") and value.endswith("}"):
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
    print("  colum")
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
