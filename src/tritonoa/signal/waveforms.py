import numpy as np
from scipy.signal import chirp


def generate_chirp(dur: float, f0: float, f1: float, fs: float = 1.0, **kwargs) -> np.ndarray:
    n_samples = int(dur * fs)
    t = np.arange(0, n_samples) / fs
    return chirp(t, f0, dur, f1, **kwargs)


def generate_cw(dur: float, fc: float, fs: float = 1.0) -> np.ndarray:
    n_samples = int(dur * fs)
    t = np.arange(0, n_samples) / fs
    return np.sin(2 * np.pi * fc * t)


def generate_gaussian_pulse(dur: float, fc: float, fs: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    n_samples = int(dur * fs)
    t = np.arange(0, n_samples) / fs
    return np.exp(-0.5 * ((t - dur / 2) / sigma) ** 2) * np.cos(2 * np.pi * fc * t)