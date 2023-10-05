import numpy as np
import tensorflow as tf


def get_spectrogram(wav: np.ndarray) -> tf.Tensor:
    wav = tf.cast(wav, dtype=tf.float32)
    spec = tf.signal.stft(wav, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    spec = spec[..., tf.newaxis]
    return spec


def squeeze_wav(wav: tf.TensorSpec, label: tf.TensorSpec) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    wav_squeezed = tf.squeeze(wav, axis=-1)
    return wav_squeezed, label


def get_spec_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(map_func=lambda wav, label: (get_spectrogram(wav), label), num_parallel_calls=tf.data.AUTOTUNE)
