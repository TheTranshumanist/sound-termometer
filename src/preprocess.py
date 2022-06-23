from typing import Union
import numpy as np
import tensorflow as tf


def get_spectrogram(wav: np.ndarray) -> tf.Tensor:
    wav = tf.cast(wav, dtype=tf.float32)
    spec = tf.signal.stft(wav, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = spec[..., tf.newaxis]
    return spec


def get_labeled_spectrogram_class(wav: np.ndarray, wav_label: str, labels: np.ndarray) -> tuple[tf.Tensor, int]:
    spec = get_spectrogram(wav)
    label = tf.argmax(wav_label == labels)
    return spec, label


def get_labeled_spectrogram_reg(wav: np.ndarray, wav_label: Union[int, float], labels: np.ndarray) -> tuple[tf.Tensor, Union[int, float]]:
    spec = get_spectrogram(wav)
    return spec, wav_label
