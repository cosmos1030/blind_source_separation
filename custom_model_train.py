import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
torchaudio.set_audio_backend("soundfile")


root_dir    = 'MiniLibriMix/train'
n_fft       = 1024
hop_length  = 256
batch_size  = 4
lr          = 1e-3
num_epochs  = 20
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

class SpeechMixDataset(Dataset):
    def __init__(self, root_dir, n_fft, hop_length):
        self.mix_dir = os.path.join(root_dir, 'mix_clean')
        self.s1_dir = os.path.join(root_dir, 's1')
        self.s2_dir = os.path.join(root_dir, 's2')
        self.files = sorted(os.listdir(self.mix_dir))
        self.n_fft = n_fft
        self.hop = hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        mix_wav, _ = torchaudio.load(os.path.join(self.mix_dir, fname))
        s1_wav, _  = torchaudio.load(os.path.join(self.s1_dir, fname))
        s2_wav, _  = torchaudio.load(os.path.join(self.s2_dir, fname))

        mix = mix_wav.mean(dim=0)
        s1  = s1_wav.mean(dim=0)
        s2  = s2_wav.mean(dim=0)

        mix_spec = torch.stft(mix, n_fft=self.n_fft, hop_length=self.hop, return_complex=True)
        s1_spec  = torch.stft(s1,  n_fft=self.n_fft, hop_length=self.hop, return_complex=True)
        s2_spec  = torch.stft(s2,  n_fft=self.n_fft, hop_length=self.hop, return_complex=True)

        mix_mag = mix_spec.abs()
        s1_mag  = s1_spec.abs()
        s2_mag  = s2_spec.abs()

        return mix_mag, s1_mag, s2_mag

def pad_collate(batch):
    mix_list, s1_list, s2_list = zip(*batch)
    T_max = max(mix.shape[1] for mix in mix_list)
    def pad(tensor):
        pad_amt = T_max - tensor.shape[1]
        return F.pad(tensor, (0, pad_amt))
    mix_batch = torch.stack([pad(m) for m in mix_list], dim=0)
    s1_batch  = torch.stack([pad(s) for s in s1_list], dim=0)
    s2_batch  = torch.stack([pad(s) for s in s2_list], dim=0)
    return mix_batch, s1_batch, s2_batch

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
        z = self.encoder(x)
        masks = self.decoder(z)
        return masks

dataset = SpeechMixDataset(root_dir, n_fft, hop_length)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    drop_last=True, collate_fn=pad_collate)

model = MaskNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for mix_mag, s1_mag, s2_mag in tqdm(loader):
        mix_mag = mix_mag.unsqueeze(1).to(device)
        s1_mag  = s1_mag.to(device)
        s2_mag  = s2_mag.to(device)

        masks = model(mix_mag) 
        e1 = masks[:,0] * mix_mag.squeeze(1)
        e2 = masks[:,1] * mix_mag.squeeze(1)

        loss = criterion(e1, s1_mag) + criterion(e2, s2_mag)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'masknet.pth')
print("Training complete. Model saved as masknet.pth")
