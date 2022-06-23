__version__ = '0.1.0'

from load import get_waveforms_class, get_waveforms_reg
from preprocess import get_labeled_spectrogram_class, get_labeled_spectrogram_reg
from augment import add_white_noise, shift_sound, stretch_sound
