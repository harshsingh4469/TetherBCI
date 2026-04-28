import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

def generate_multimodal_data(n_samples=200):
    os.makedirs("data/eeg", exist_ok=True)
    os.makedirs("data/fmri", exist_ok=True)
    os.makedirs("data/meg", exist_ok=True)
    print(f"Generating {n_samples} multimodal brain samples...")
    for i in range(n_samples):
        label = i % 4
        eeg   = np.random.randn(64, 256).astype(np.float32)
        t     = np.linspace(0, 1, 256)
        freq  = [10, 20, 30, 40][label]
        for ch in range(64):
            eeg[ch] += np.sin(2 * np.pi * freq * t) * 0.5
        eeg = (eeg - eeg.mean()) / (eeg.std() + 1e-8)
        fmri = np.zeros((64, 64), dtype=np.float32)
        cx, cy = np.random.randint(15, 50, 2)
        for x in range(64):
            for y in range(64):
                fmri[x,y] += np.exp(-((x-cx)**2 + (y-cy)**2) / (2*10**2))
        fmri = (fmri - fmri.min()) / (fmri.max() - fmri.min() + 1e-8)
        meg = np.random.randn(306, 256).astype(np.float32)
        for ch in range(306):
            meg[ch] += np.sin(2 * np.pi * freq * t) * 0.3
        meg = (meg - meg.mean()) / (meg.std() + 1e-8)
        np.save(f"data/eeg/sample_{i:03d}_label{label}.npy",  eeg)
        np.save(f"data/fmri/sample_{i:03d}_label{label}.npy", fmri)
        np.save(f"data/meg/sample_{i:03d}_label{label}.npy",  meg)
        if (i+1) % 50 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")
    print("Done! Data saved in data/")

class MultiModalBCIDataset(Dataset):
    def __init__(self, eeg_dir, fmri_dir, meg_dir):
        self.eeg_files  = sorted([os.path.join(eeg_dir,  f) for f in os.listdir(eeg_dir)  if f.endswith('.npy')])
        self.fmri_files = sorted([os.path.join(fmri_dir, f) for f in os.listdir(fmri_dir) if f.endswith('.npy')])
        self.meg_files  = sorted([os.path.join(meg_dir,  f) for f in os.listdir(meg_dir)  if f.endswith('.npy')])
        print(f"Found {len(self.eeg_files)} multimodal samples")

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        eeg   = torch.FloatTensor(np.load(self.eeg_files[idx]))
        fmri  = torch.FloatTensor(np.load(self.fmri_files[idx])).unsqueeze(0)
        meg   = torch.FloatTensor(np.load(self.meg_files[idx]))
        label = int(self.eeg_files[idx].split('label')[1].split('.')[0])
        return eeg, fmri, meg, label

if __name__ == "__main__":
    generate_multimodal_data()