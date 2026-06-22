#!/usr/bin/env python3
# Bulgarian-localized Linux version
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

# Състояния
left_hold_active = False
left_click_loop_active = False

# Аудио настройки
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

# Модел
MODEL_NAME = "medium"      # small / medium
DEVICE = "cpu"             # cpu / cuda
COMPUTE_TYPE = "int8"      # int8 for CPU, float16 for CUDA usually

# Задаване на български като предпочитан език
FORCE_LANGUAGE = "bg"      # None = auto detect, "bg" = Bulgarian, "en" = English

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

def hold_left_mouse_until_escape() -> None:
    global left_hold_active
    if left_hold_active:
        print("[INFO] Вече задържате левия бутон.")
        return
    left_hold_active = True
    print("[МИШКА] Задържам ЛЯВ бутон. Натиснете ESC за освобождаване.")
    try:
        pyautogui.mouseDown(button="left")
        keyboard.wait("esc")
    finally:
        pyautogui.mouseUp(button="left")
        left_hold_active = False
        print("[МИШКА] ЛЯВ бутон освободен.")

def click_left_loop_until_escape(interval_seconds: float = 0.08) -> None:
    global left_click_loop_active
    if left_click_loop_active:
        print("[INFO] Вече работи циклично кликване.")
        return
    left_click_loop_active = True
    print("[МИШКА] Циклично кликване (ЛЯВО) стартирано. Натиснете ESC за стоп.")
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
        print("[МИШКА] Цикличното кликване спря.")

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
        "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
        "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
        "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
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
    if first in {"go", "write", "пиши", "запиши", "напиши", "попълни"}:
        return " ".join(words[1:]).strip()
    return None

def wait_for_speech_and_record():
    frames = []
    speech_started = False
    speech_duration = 0.0
    silence_after_speech = 0.0
    started_at = None
    print("[ЧАКАНЕ] Очаквам реч...")
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
                    print("[ON] Реч засечена. Слушам...")
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
                print("[INFO] Достигнат максимален период за фраза.")
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
        "спри", "престани", "спри слушането", "изход", "излез", "край", "стоп"
    }
    if normalized in stop_commands:
        print("[STOP] Получена команда за изход.")
        return True
    if normalized in {"задръж ляво", "задръж ляв", "hold left", "left hold"}:
        threading.Thread(target=hold_left_mouse_until_escape, daemon=True).start()
        return False
    if normalized in {"спам ляв", "клик ляв кръг", "keep left", "click left", "left click loop"}:
        threading.Thread(target=click_left_loop_until_escape, daemon=True).start()
        return False
    # Командни карти (български + английски)
    command_map = {
        "enter": ("key", "enter"),
        "нова линия": ("key", "enter"),
        "tab": ("key", "tab"),
        "backspace": ("key", "backspace"),
        "delete": ("key", "delete"),
        "запетая": ("text", ","),
        "точка": ("text", "."),
        "въпросителен знак": ("text", "?"),
        "възклицателен знак": ("text", "!"),
        "двоеточие": ("text", ":"),
        "точка и запетая": ("text", ";"),
        "горе": ("key", "up"),
        "долу": ("key", "down"),
        "наляво": ("key", "left"),
        "надясно": ("key", "right"),
        "page down": ("key", "page down"),
        "page up": ("key", "page up"),
        "вертикална черта": ("text", "|"),
    }
    shortcut_map = {
        "избери всичко": ("ctrl", "a"),
        "копирай": ("ctrl", "c"),
        "постави": ("ctrl", "v"),
        "изрежи": ("ctrl", "x"),
        "сменя прозорец": ("alt", "tab"),
        "затвори прозорец": ("alt", "f4"),
    }
    go_part = extract_go_remainder(raw)
    if go_part is not None:
        if go_part:
            print(f"[ТЕКСТ] Пиша: {go_part}")
            type_text(go_part, add_trailing_space=True)
        else:
            print("[INFO] 'go' / 'пиши' беше казано без текст.")
        return False
    if normalized in shortcut_map:
        print(f"[КОМАНДА] Изпълнявам: {normalized}")
        send_hotkey(*shortcut_map[normalized])
        return False
    if normalized in command_map:
        kind, value = command_map[normalized]
        print(f"[КОМАНДА] Изпълнявам: {normalized}")
        if kind == "key":
            send_special_key(value)
        else:
            type_text(value, add_trailing_space=False)
        return False
    print("[ОБРЪЩЕНИЕ] Не е разпозната команда. За писане използвайте: go <текст> или пиши <текст>")
    return False

def process_transcribed_text(text: str) -> bool:
    return handle_command_or_text(text)

def main() -> None:
    print("Зареждам Whisper модела...")
    print(f"Модел: {MODEL_NAME}, device={DEVICE}, compute_type={COMPUTE_TYPE}")
    print("При първо изпълнение моделът може да бъде изтеглен автоматично.\n")
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    try:
        print("Налични аудио устройства:")
        print(sd.query_devices())
        print()
    except Exception as e:
        print(f"Не мога да прочета аудио устройствата: {e}\n")
    print("Готово.")
    print("Командите се изпълняват директно, без ключова дума за активиране.")
    print("За писане използвайте: go <текст> или пиши <текст>")
    print("Примери:")
    print("  избери всичко")
    print("  горе")
    print("  page down")
    print("  колонa")
    print("  go здравей свят")
    print("  пиши това е текст")
    print("За изход: Ctrl+C или кажете 'спри' / 'изход'\n")
    try:
        while True:
            wav_path = wait_for_speech_and_record()
            if wav_path is None:
                print("[OFF] Няма валидна фраза.\n")
                continue
            try:
                text, detected_language = transcribe_file(model, wav_path)
            finally:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
            if not text:
                print("[OFF] Нищо не беше разпознато.\n")
                continue
            if detected_language:
                print(f"[ЕЗИК={detected_language}] {text}")
            else:
                print(f"[ТЕКСТ] {text}")
            should_stop = process_transcribed_text(text)
            print("[OFF] Готово. Очаквам нова реч.\n")
            if should_stop:
                break
    except KeyboardInterrupt:
        print("\nСпряно от потребителя.")
    except Exception as e:
        print(f"\nФатална грешка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
