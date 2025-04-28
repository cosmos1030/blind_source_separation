import os
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
from scipy.special import expn
from sklearn.decomposition import FastICA

root_dir = 'MiniLibriMix/val'
mix_dir  = os.path.join(root_dir, 'mix_clean')
s1_dir   = os.path.join(root_dir, 's1')
s2_dir   = os.path.join(root_dir, 's2')
out_dir1 = 'predictions_ica/s1'
out_dir2 = 'predictions_ica/s2'
os.makedirs(out_dir1, exist_ok=True)
os.makedirs(out_dir2, exist_ok=True)

def logmmse_enhance(y, sr, n_fft=1024, hop_length=None, win_length=None,
                    noise_frames=6, alpha=0.98, eps=1e-10):
    if hop_length is None:
        hop_length = n_fft // 2
    if win_length is None:
        win_length = n_fft
    window = np.hanning(win_length)
    f, t, Zxx = stft(y, fs=sr, window=window,
                     nperseg=win_length, noverlap=win_length-hop_length)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    noise_psd = np.mean(mag[:, :noise_frames]**2, axis=1) + eps
    enhanced_mag = np.zeros_like(mag)
    prev_frame = np.zeros_like(noise_psd)
    for i in range(mag.shape[1]):
        power = mag[:, i]**2
        gamma = power / noise_psd
        if i == 0:
            xi = np.maximum(gamma - 1, 0)
        else:
            xi = alpha * (prev_frame**2) / noise_psd + (1 - alpha) * np.maximum(gamma - 1, 0)
            xi = np.maximum(xi, 0)
        v = xi * gamma / (1 + xi)
        v = np.maximum(v, eps)
        gain = (xi / (1 + xi)) * np.exp(0.5 * expn(1, v))
        gain = np.nan_to_num(gain, nan=(xi/(1+xi)), posinf=(xi/(1+xi)), neginf=0.0)
        enhanced_mag[:, i] = gain * mag[:, i]
        prev_frame = enhanced_mag[:, i]
        noise_psd = alpha * noise_psd + (1 - alpha) * power
    Zxx_enh = enhanced_mag * np.exp(1j * phase)
    _, y_enh = istft(Zxx_enh, fs=sr, window=window,
                     nperseg=win_length, noverlap=win_length-hop_length,
                     input_onesided=True)
    return y_enh[:len(y)]

def ica_separate(y_orig, y_est):
    X = np.vstack((y_est, y_orig)).T
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ica = FastICA(n_components=2, random_state=0, max_iter=500, tol=1e-4)
    S_ = ica.fit_transform(X)
    return S_[:, 0], S_[:, 1]

def si_snr(est, ref, eps=1e-8):
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)
    proj = np.sum(est_zm * ref_zm) * ref_zm / (np.sum(ref_zm**2) + eps)
    noise = est_zm - proj
    return 10 * np.log10((np.sum(proj**2) + eps) / (np.sum(noise**2) + eps))

snr_scores = []
files = sorted(os.listdir(mix_dir))
for fname in files:
    mix, sr = sf.read(os.path.join(mix_dir, fname))
    ref1, _ = sf.read(os.path.join(s1_dir, fname))
    ref2, _ = sf.read(os.path.join(s2_dir, fname))

    mix_est = logmmse_enhance(mix, sr)

    s1_est, s2_est = ica_separate(mix, mix_est)

    s1_final = logmmse_enhance(s1_est, sr)
    s2_final = logmmse_enhance(s2_est, sr)

    L = min(len(ref1), len(ref2), len(s1_final), len(s2_final))
    ref1, ref2 = ref1[:L], ref2[:L]
    s1_final, s2_final = s1_final[:L], s2_final[:L]

    snr_a = (si_snr(s1_final, ref1) + si_snr(s2_final, ref2)) / 2
    snr_b = (si_snr(s1_final, ref2) + si_snr(s2_final, ref1)) / 2
    best_snr = max(snr_a, snr_b)
    snr_scores.append(best_snr)

    sf.write(os.path.join(out_dir1, fname), s1_final, sr)
    sf.write(os.path.join(out_dir2, fname), s2_final, sr)

print(f"Permutation-invariant Average SI-SNR: {np.mean(snr_scores):.2f} dB over {len(snr_scores)} files")
