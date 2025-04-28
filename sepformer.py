import os
import torch
import torchaudio
import numpy as np
from speechbrain.pretrained import SepformerSeparation as separator

root_dir = 'MiniLibriMix/val'
mix_dir = os.path.join(root_dir, 'mix_clean')
s1_dir = os.path.join(root_dir, 's1')
s2_dir = os.path.join(root_dir, 's2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def si_snr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = torch.sum(est * ref) * ref / (torch.sum(ref**2) + eps)
    noise = est - proj
    return 10 * torch.log10((torch.sum(proj**2) + eps) / (torch.sum(noise**2) + eps))

model = separator.from_hparams(
    source="speechbrain/sepformer-libri2mix",
    savedir="pretrained_models/sepformer-libri2mix",
    run_opts={"device": device}
)
model.eval()

os.makedirs('predictions_sepformer_sb/s1', exist_ok=True)
os.makedirs('predictions_sepformer_sb/s2', exist_ok=True)

snr_scores = []

for fname in sorted(os.listdir(mix_dir)):
    mix_wav, sr = torchaudio.load(os.path.join(mix_dir, fname))
    s1_wav, _   = torchaudio.load(os.path.join(s1_dir, fname))
    s2_wav, _   = torchaudio.load(os.path.join(s2_dir, fname))

    if sr != 8000:
        resamp = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
        mix_wav = resamp(mix_wav)
        s1_wav  = resamp(s1_wav)
        s2_wav  = resamp(s2_wav)
        sr = 8000
    mix_wav = mix_wav.mean(dim=0, keepdim=True).to(device)

    with torch.no_grad():
        est_sources = model.separate_batch(mix_wav)  # (batch=1, time, 2)
    est1 = est_sources[0, :, 0].cpu()
    est2 = est_sources[0, :, 1].cpu()

    ref1 = s1_wav.mean(dim=0)[:est1.shape[-1]]
    ref2 = s2_wav.mean(dim=0)[:est2.shape[-1]]

    L = min(est1.numel(), ref1.numel(), est2.numel(), ref2.numel())
    est1, est2 = est1[:L], est2[:L]
    ref1, ref2 = ref1[:L], ref2[:L]

    snr_11 = si_snr(est1, ref1)
    snr_22 = si_snr(est2, ref2)
    snr_a = (snr_11 + snr_22) / 2

    snr_12 = si_snr(est1, ref2)
    snr_21 = si_snr(est2, ref1)
    snr_b = (snr_12 + snr_21) / 2

    best_snr = max(snr_a, snr_b).item()
    snr_scores.append(best_snr)

    torchaudio.save(os.path.join('predictions_sepformer_sb/s1', fname),
                    est1.unsqueeze(0), sample_rate=sr)
    torchaudio.save(os.path.join('predictions_sepformer_sb/s2', fname),
                    est2.unsqueeze(0), sample_rate=sr)

print(f"Average SI-SNR: {np.mean(snr_scores):.2f} dB over {len(snr_scores)} files")
print("Outputs saved in predictions_sepformer_sb/s1 and /s2")
