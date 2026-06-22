Overview

lazy.py is a Windows-focused voice control and voice typing script that listens to microphone input, transcribes speech with Faster-Whisper, and then either types text or triggers keyboard and mouse actions based on spoken commands.

The script records audio from the default input device, detects when speech starts and ends using volume thresholds, saves the captured phrase as a temporary WAV file, and sends it to a Whisper model for transcription.
What the script does

The script works as a lightweight speech-to-command bridge for desktop use on Windows.

It supports these main behaviors:

    Voice typing with the go <text> or пиши <text> pattern, which types the spoken text into the active window.

    Special key commands such as enter, tab, backspace, delete, arrow keys, page up, and page down.

    Shortcut commands such as copy, paste, select all, ctrl a, alt tab, function keys, and escape.

    Mouse automation, including holding the left mouse button until Esc is pressed and running a repeated left-click loop until Esc is pressed.

    Stop commands such as stop, exit, quit, край, and спри, which terminate the program loop.

How it works

The script initializes a WhisperModel from faster_whisper using configurable values for model name, device, compute type, and forced language.

Audio is captured with sounddevice at 16 kHz, mono, in short blocks, and the script computes RMS audio level with NumPy to decide when speech has started and when silence is long enough to stop recording.

After transcription, the recognized text is normalized and matched against command maps or shortcut maps. If the text starts with go or пиши, the remainder is typed into the active application using PowerShell System.Windows.Forms.SendKeys.

Mouse actions are performed through win32api and win32con, while hotkey detection and stop-on-escape behavior are handled through the keyboard package.
Platform requirements

This script is designed for Windows, not Linux or macOS, because it depends on win32api, win32con, PowerShell SendKeys, and Windows-style keyboard and mouse event handling.

A working microphone and an accessible default audio input device are also required for normal operation.
Python packages to install

The script imports both standard-library modules and third-party packages.

Third-party packages required with pip:

    numpy

    sounddevice

    faster-whisper

    keyboard

    pywin32
For Windows:

    python -m pip install faster-whisper sounddevice numpy keyboard pywin32

For Linux:

    pip install numpy sounddevice faster-whisper keyboard pyautogui
    apt install ffmpeg python3-xlib

Example installation command:

bash
pip install numpy sounddevice faster-whisper keyboard pywin32

System-level notes

faster-whisper relies on the CTranslate2 runtime and may download the selected Whisper model on first run, which the script also notes in its startup messages.

For sounddevice, the system must have a working PortAudio-compatible audio setup, and on Windows this usually works directly when Python audio dependencies are installed correctly.

If DEVICE is changed from cpu to cuda, a CUDA-capable environment and compatible GPU stack are needed for hardware acceleration.
Main configuration values

The script exposes several constants near the top that control behavior:
Setting Purpose
Setting Purpose
SAMPLE_RATE     Audio capture rate, set to 16000 Hz.
CHANNELS        Mono recording, set to 1.
START_THRESHOLD Audio level needed to start recording speech.
CONTINUE_THRESHOLD      Lower threshold used to continue an active recording.
SILENCE_HOLD_SECONDS    Silence duration required to finish a phrase.
MIN_SPEECH_SECONDS      Minimum speech length for a valid phrase.
MAX_RECORD_SECONDS      Maximum duration for one recorded phrase.
MODEL_NAME      Whisper model size, currently medium.
DEVICE  Inference device, currently cpu.
COMPUTE_TYPE    Inference precision, currently int8.
FORCE_LANGUAGE  Forced recognition language, currently en.
Typical usage

Run the script in a Windows terminal while the target application is focused or ready to receive keyboard input.

Examples of supported phrases from the script include go hello world, select all, up, page down, colum, and the Bulgarian form пиши това е текст.

Because the script sends keystrokes to the active window, it should be used carefully in applications where unintended input could have side effects.
Limitations

The script is single-process and stateful around active mouse actions, so misrecognition can trigger input in whichever window currently has focus.

It also uses direct string-based command matching rather than intent parsing, which means command reliability depends heavily on the transcription quality and the exact spoken phrases expected by the mapping tables.
