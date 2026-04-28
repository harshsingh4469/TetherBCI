import torch
import torch.nn as nn


# --- Mental State Decoder ---
class MentalStateDecoder(nn.Module):
    """
    Decodes latent brain representation into mental state classification.
    Predicts: rest, motor imagery, visual, cognitive (4 classes)
    """
    def __init__(self, latent_dim=256, n_classes=4):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, n_classes)
        )

    def forward(self, z):
        return self.decoder(z)  # (B, n_classes)


# --- EEG Signal Decoder ---
class EEGSignalDecoder(nn.Module):
    """
    Decodes latent vector back into EEG signal.
    Used for self-supervised learning and signal reconstruction.
    """
    def __init__(self, latent_dim=256, n_channels=64, seq_len=256):
        super().__init__()

        self.project = nn.Linear(latent_dim, 512 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),

            nn.ConvTranspose1d(64, n_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):               # z: (B, latent_dim)
        x = self.project(z)             # (B, 512*8)
        x = x.view(-1, 512, 8)         # (B, 512, 8)
        return self.decoder(x)          # (B, 64, 128) ≈ EEG signal


# --- fMRI Image Decoder ---
class fMRIDecoder(nn.Module):
    """
    Decodes latent vector into fMRI brain image.
    Used for cross-modal synthesis (EEG → fMRI).
    """
    def __init__(self, latent_dim=256):
        super().__init__()

        self.project = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),   # 8x8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),   # 16x16

            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.GELU(),   # 32x32

            nn.ConvTranspose2d(64,  1,   kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()                       # 64x64
        )

    def forward(self, z):                       # z: (B, latent_dim)
        x = self.project(z)                     # (B, 512*4*4)
        x = x.view(-1, 512, 4, 4)              # (B, 512, 4, 4)
        return self.decoder(x)                  # (B, 1, 64, 64)