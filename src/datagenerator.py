from typing import Union, Callable
import numpy as np
import tensorflow as tf
import src


AUTOTUNE = tf.data.AUTOTUNE


class DataGenerator:

    dataset: tf.data.Dataset
    wav_list: list[tuple[np.ndarray, Union[str, float, int]]]
    labels: np.ndarray
    is_class: bool

    def __init__(self,
                 files: list[str],
                 labels: np.ndarray,
                 augment_fns: list[Callable[[np.ndarray,
                                             Union[float, int]], np.ndarray]] = [],
                 augment_args: dict[Callable[[np.ndarray, Union[float, int]],
                                             np.ndarray], Union[float, int]] = {},
                 is_class=True) -> None:
        self.labels = labels

        if is_class:
            self.wav_list = src.get_waveforms_class(files)
        else:
            self.wav_list = src.get_waveforms_reg(files)
            
        self.wav_list = self.__augment(augment_fns, augment_args)
        self.dataset = self.__preprocess()
        
        

    def __augment(self,
                  fns: list[Callable[[np.ndarray, Union[float, int]], np.ndarray]],
                  args: dict[Callable[[np.ndarray, Union[float, int]],
                                      np.ndarray], Union[float, int]]) -> list[tuple[np.ndarray, Union[str, float, int]]]:
        new_wav_list = []
        for wav, label in self.wav_list:
            new_wav_list.append((wav, label))
            for fn in fns:
                aug_wav = fn(wav, args[fn])
                new_wav_list.append(aug_wav, label)
        return new_wav_list

    def __preprocess(self) -> tf.data.Dataset:
        wav_list, wav_labels = zip(**self.wav_list)
        ds = tf.data.Dataset.from_tensor_slices((wav_list, wav_labels))
        
        if self.is_class:
            return ds.map(src.get_labeled_spectrogram_class, num_parallel_calls=AUTOTUNE)
        return ds.map(src.get_labeled_spectrogram_reg, num_parallel_calls=AUTOTUNE)
            
        