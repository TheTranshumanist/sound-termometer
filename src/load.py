from typing import Union
import librosa
import os
import numpy as np

INPUT_LEN = 16000

def get_file_label(file_path: str) -> str:
    return os.path.basename(os.path.dirname(file_path))


def get_waveforms_class(file_list: list[str]) -> list[tuple[np.ndarray, str]]:
    waveforms = []
    for file in file_list:
        wav = librosa.load(file)[0]
        label = get_file_label(file)
        
        if len(wav) > INPUT_LEN:
            wav = wav[:INPUT_LEN]
        else:
            wav = np.pad(wav, (0, max(0, INPUT_LEN - len(wav))), 'constant')
        
        waveforms.append((wav, label))
    return waveforms

def get_waveforms_reg(file_list: list[str]) -> list[tuple[np.ndarray, Union[int, float]]]:
    pass