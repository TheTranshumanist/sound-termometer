import librosa
import numpy as np

INPUT_LEN = 16000


def add_white_noise(wav: np.ndarray, amplitude: float = 0.005) -> np.ndarray:
    wn = np.random.randn(len(wav))
    return wav + wn * amplitude


def shift_sound(wav: np.ndarray, shift: float = 1600) -> np.ndarray:
    return np.roll(wav, shift)


def stretch_sound(wav: np.ndarray, rate: float = 1) -> np.ndarray:
    wav = librosa.effects.time_stretch(wav, rate)
    if len(wav) > INPUT_LEN:
        wav = wav[:INPUT_LEN]
    else:
        wav = np.pad(wav, (0, max(0, INPUT_LEN - len(wav))), 'constant')
    return wav
