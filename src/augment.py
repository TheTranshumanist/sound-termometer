import librosa
import numpy as np
import tensorflow as tf

INPUT_LEN = 16000


def add_white_noise(wav: tf.TensorSpec, label: tf.TensorSpec,
                    amplitude: float = 0.005) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    wav_noise = np.random.randn(wav.shape[-1])
    return tf.convert_to_tensor(wav.numpy() + wav_noise * amplitude), label


def shift_sound(wav: tf.TensorSpec, label: tf.TensorSpec, shift: float = 1600.0) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    wav_roll = np.roll(wav.numpy(), int(shift))
    return tf.convert_to_tensor(wav_roll), label


def stretch_sound(wav: tf.TensorSpec, label: tf.TensorSpec, rate: float = 1.0) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    wav_stretched = librosa.effects.time_stretch(wav.numpy(), rate)
    if len(wav_stretched) > INPUT_LEN:
        wav_stretched = wav_stretched[:INPUT_LEN]
    else:
        wav_stretched = np.pad(wav_stretched, (0, max(0, INPUT_LEN - len(wav_stretched))), 'constant')
    return tf.convert_to_tensor(wav_stretched), label
