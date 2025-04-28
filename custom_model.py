import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# --- Hyperparameters ---
root_dir    = 'MiniLibriMix/val'  
n_fft       = 1024
hop_length  = 256
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Hann windows ---
window_cpu = torch.hann_window(n_fft)
window_gpu = torch.hann_window(n_fft, device=device)

# --- SI-SNR evaluation ---
def si_snr(est, ref, eps=1e-8):
    est_zm = est - est.mean()
    ref_zm = ref - ref.mean()
    proj = torch.sum(est_zm * ref_zm) * ref_zm / (torch.sum(ref_zm**2) + eps)
    noise = est_zm - proj
    return 10 * torch.log10((torch.sum(proj**2) + eps) / (torch.sum(noise**2) + eps))

# --- Validation Dataset ---
class SpeechMixValDataset(Dataset):
    def __init__(self, root_dir, n_fft, hop_length):
        self.mix_dir = os.path.join(root_dir, 'mix_clean')
        self.s1_dir  = os.path.join(root_dir, 's1')
        self.s2_dir  = os.path.join(root_dir, 's2')
        self.files   = sorted(os.listdir(self.mix_dir))
        self.n_fft   = n_fft
        self.hop     = hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        mix_wav, sr = torchaudio.load(os.path.join(self.mix_dir, fname))
        s1_wav, _   = torchaudio.load(os.path.join(self.s1_dir,  fname))
        s2_wav, _   = torchaudio.load(os.path.join(self.s2_dir,  fname))

        mix = mix_wav.mean(dim=0)
        s1  = s1_wav.mean(dim=0)
        s2  = s2_wav.mean(dim=0)

        mix_spec = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop,
                              window=window_cpu, return_complex=True)
        phase    = torch.angle(mix_spec)
        mix_mag  = mix_spec.abs()

        return mix_mag, phase, s1, s2, sr, fname

# --- Model ---
class MaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Load model ---
model = MaskNet().to(device)
model.load_state_dict(torch.load('masknet.pth', map_location=device))
model.eval()

# --- DataLoader ---
dataset = SpeechMixValDataset(root_dir, n_fft, hop_length)
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Output dirs ---
os.makedirs('predictions_custom/s1', exist_ok=True)
os.makedirs('predictions_custom/s2', exist_ok=True)

# --- Inference & Permutation-Invariant Evaluation ---
snr_scores = []
for mix_mag, phase, s1, s2, sr, fname in loader:
    mix_mag = mix_mag.unsqueeze(1).to(device)  # [1,1,F,T]
    phase   = phase.to(device)

    with torch.no_grad():
        masks = model(mix_mag)         # [1,2,F,T]
    m1, m2    = masks[:,0], masks[:,1]

    Zxx       = mix_mag.squeeze(1) * torch.exp(1j * phase)
    est1_spec = m1 * Zxx
    est2_spec = m2 * Zxx

    x1 = torch.istft(est1_spec, n_fft=n_fft, hop_length=hop_length,
                     window=window_gpu, length=s1.shape[-1])
    x2 = torch.istft(est2_spec, n_fft=n_fft, hop_length=hop_length,
                     window=window_gpu, length=s2.shape[-1])

    # Align lengths
    L = min(x1.numel(), x2.numel(), s1.numel(), s2.numel())
    est1 = x1.cpu().squeeze(0)[:L]
    est2 = x2.cpu().squeeze(0)[:L]
    ref1 = s1[:L]
    ref2 = s2[:L]

    # Permutation-invariant SI-SNR
    snr_a = (si_snr(est1, ref1) + si_snr(est2, ref2)) / 2
    snr_b = (si_snr(est1, ref2) + si_snr(est2, ref1)) / 2
    if snr_b > snr_a:
        best_snr = snr_b.item()
        # swap for correct saving
        est1, est2 = est2, est1
    else:
        best_snr = snr_a.item()

    snr_scores.append(best_snr)

    # Save 2D tensors [1, time]
    torchaudio.save(os.path.join('predictions_custom/s1', fname[0]),
                    est1.unsqueeze(0), sample_rate=sr)
    torchaudio.save(os.path.join('predictions_custom/s2', fname[0]),
                    est2.unsqueeze(0), sample_rate=sr)

# --- Final report ---
avg_snr = np.mean(snr_scores)
print(f"Permutation-invariant SI-SNR per file: {snr_scores}")
print(f"Average SI-SNR: {avg_snr:.2f} dB")

