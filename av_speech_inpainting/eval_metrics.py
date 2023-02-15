import subprocess
import re
import numpy as np
from scipy.signal import stft
from mir_eval.separation import bss_eval_sources


def l1_eval(target, estimated, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):
    window_frame_len = int(window_size / 1e3 * sample_rate)
    overlap_frame_len = int((window_size - step_size) / 1e3 * sample_rate)
    
    sample_len = min([len(target), len(estimated)])
    target = target[: sample_len]
    estimated = estimated[: sample_len]
    
    freqs, times, target_stft = stft(target, nperseg=window_frame_len, noverlap=overlap_frame_len, nfft=n_fft)
    freqs, times, estimated_stft = stft(estimated, nperseg=window_frame_len, noverlap=overlap_frame_len, nfft=n_fft)
    target_spec = np.log(np.abs(target_stft) + 1e-6)
    estimated_spec = np.log(np.abs(estimated_stft) + 1e-6)
    
    l1_error = np.abs(target_spec - estimated_spec).sum()

    return l1_error


def l2_eval(target, estimated, sample_rate=16e3, n_fft=512, window_size=25, step_size=10):
    window_frame_len = int(window_size / 1e3 * sample_rate)
    overlap_frame_len = int((window_size - step_size) / 1e3 * sample_rate)
    
    sample_len = min([len(target), len(estimated)])
    target = target[: sample_len]
    estimated = estimated[: sample_len]
    
    freqs, times, target_stft = stft(target, nperseg=window_frame_len, noverlap=overlap_frame_len, nfft=n_fft)
    freqs, times, estimated_stft = stft(estimated, nperseg=window_frame_len, noverlap=overlap_frame_len, nfft=n_fft)
    target_spec = np.log(np.abs(target_stft) + 1e-6)
    estimated_spec = np.log(np.abs(estimated_stft) + 1e-6)
    
    l2_error = np.square(target_spec - estimated_spec).sum()

    return l2_error


def sdr_eval(target, estimated):
     sample_len = min([len(target), len(estimated)])
     target = target[: sample_len]
     estimated = estimated[: sample_len]
     
     # Skip evaluation if estimated is all-zero vector
     if np.any(estimated):
         sdr, sir, sar, _ = bss_eval_sources(np.array([target]), np.array([estimated]), compute_permutation=False)
         return sdr[0], sir[0], sar[0]
     else:
         return np.nan, np.nan, np.nan


def sisdr_eval(ref_sig, out_sig, eps=1e-8):
    """Calculate Scale-Invariant Source-to-Distortion Ratio (SI-SDR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SI-SDR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisdr = 10 * np.log10(ratio + eps)
    
    return sisdr


def pesq_eval(source_file_path, estimation_file_path, pesq_bin_path, mode='wb'):
    if mode == 'nb':
        command_args = [pesq_bin_path, '+16000', source_file_path, estimation_file_path]
    else:
        command_args = [pesq_bin_path, '+16000', '+wb', source_file_path, estimation_file_path]
    
    try:
        output = subprocess.check_output(command_args)

        if mode == 'nb':
            match = re.search("\(Raw MOS, MOS-LQO\):\s+= (-?[0-9.]+?)\t([0-9.]+?)$", output.decode().replace('\r', ''), re.MULTILINE)
            mos = float(match.group(1))
            moslqo = float(match.group(2))
            return mos, moslqo
        else:
            match = re.search("\(MOS-LQO\):\s+= ([0-9.]+?)$", output.decode().replace('\r', ''), re.MULTILINE)
            mos = float(match.group(1))
            return mos, None
    except (subprocess.CalledProcessError, AttributeError):
        return np.nan, np.nan