import os
import torch
import torchaudio
import numpy as np
from asteroid.models import ConvTasNet

root_dir = 'MiniLibriMix/val'
mix_dir = os.path.join(root_dir, 'mix_clean')
s1_dir = os.path.join(root_dir, 's1')
s2_dir = os.path.join(root_dir, 's2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def si_snr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = torch.sum(est * ref) * ref / (torch.sum(ref ** 2) + eps)
    noise = est - proj
    return 10 * torch.log10((torch.sum(proj ** 2) + eps) / (torch.sum(noise ** 2) + eps))

model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k").to(device)
model.eval()

os.makedirs('predictions/s1', exist_ok=True)
os.makedirs('predictions/s2', exist_ok=True)

snr_scores = []

for fname in sorted(os.listdir(mix_dir)):
    mix_wav, sr = torchaudio.load(os.path.join(mix_dir, fname))
    s1_wav, _   = torchaudio.load(os.path.join(s1_dir, fname))
    s2_wav, _   = torchaudio.load(os.path.join(s2_dir, fname))

    mix_wav = mix_wav.mean(dim=0, keepdim=True).to(device)
    s1_wav = s1_wav.mean(dim=0)
    s2_wav = s2_wav.mean(dim=0)

    with torch.no_grad():
        est_sources = model.separate(mix_wav)
    est1 = est_sources[0,0].cpu()
    est2 = est_sources[0,1].cpu()

    L = min(est1.numel(), est2.numel(), s1_wav.numel(), s2_wav.numel())
    est1, est2 = est1[:L], est2[:L]
    ref1, ref2 = s1_wav[:L], s2_wav[:L]

    snr_a = (si_snr(est1, ref1) + si_snr(est2, ref2)) / 2
    snr_b = (si_snr(est1, ref2) + si_snr(est2, ref1)) / 2
    best_snr = max(snr_a, snr_b).item()
    snr_scores.append(best_snr)

    torchaudio.save(f'predictions/s1/{fname}', est1.unsqueeze(0), sr)
    torchaudio.save(f'predictions/s2/{fname}', est2.unsqueeze(0), sr)

avg_snr = np.mean(snr_scores)
print(f"Average SI-SNR (permutation-invariant): {avg_snr:.2f} dB over {len(snr_scores)} files")

