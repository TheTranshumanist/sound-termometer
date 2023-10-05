from typing import Callable
import numpy as np
import tensorflow as tf
import src


AUTOTUNE = tf.data.AUTOTUNE
AUGMENT_FUNC = Callable[[tf.TensorSpec, tf.TensorSpec, float], tuple[tf.TensorSpec, tf.TensorSpec]]


class AudioDataGenerator:

    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    labels: np.ndarray

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 validation_split: float,
                 ) -> None:
        self.train_ds, self.val_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=data_path,
            batch_size=batch_size,
            validation_split=validation_split,
            seed=0,
            output_sequence_length=16000,
            subset='both'
        )

        self.train_ds = self.train_ds.map(src.squeeze_wav, AUTOTUNE)
        self.val_ds = self.val_ds.map(src.squeeze_wav, AUTOTUNE)

        self.test_ds = self.val_ds.shard(num_shards=2, index=0)
        self.val_ds = self.val_ds.shard(num_shards=2, index=1)

    def augment(self, augment_fns: dict[AUGMENT_FUNC, float]) -> None:
        for fn, arg in augment_fns.items():
            self.train_ds.concatenate(self.train_ds.map(map_func=lambda wav, label: fn(wav, label, arg),
                                                        num_parallel_calls=AUTOTUNE))
        self.train_ds = self.train_ds.shuffle(100)

    def generate_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        train_spec_ds = src.get_spec_dataset(self.train_ds)
        val_spec_ds = src.get_spec_dataset(self.val_ds)
        test_spec_ds = src.get_spec_dataset(self.test_ds)

        return train_spec_ds, val_spec_ds, test_spec_ds
