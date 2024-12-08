
import numpy as np
import pywt


def wavelet_denoising(emg_data_input, fs, wavelet, window_size=10, level=5):
    emg_data = emg_data_input.copy()
    print("=====================")
    print(f"Denoising signal using {wavelet} Wavelet ...")
    for source in range(len(emg_data)):
        for i in range(0, len(emg_data[source]) - window_size * fs, window_size * fs):
            signal = emg_data[source, i:i + window_size * fs]
            coefficients = pywt.wavedec(signal, wavelet, level)
            for j in range(1, len(coefficients)):
                coefficients[j] = pywt.threshold(coefficients[j], np.std(coefficients[j]))
            denoised_signal = pywt.waverec(coefficients, wavelet, level)
            emg_data[source, i:i + window_size * fs] = denoised_signal
    return emg_data


def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean


def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T)) / n


def whiten(emg_signal):
    whiteM = get_whiten_matrix(emg_signal)
    emg_signal = np.dot(whiteM, emg_signal)
    return emg_signal

def get_whiten_matrix(emg_signal):
    cov_matrix = covariance(emg_signal)
    U, S, V = np.linalg.svd(cov_matrix)
    d = np.diag(1.0 / np.sqrt(S))
    whiteM = np.dot(U, np.dot(d, U.T))
    return whiteM

