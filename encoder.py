import torch
import torch.nn as nn


# --- EEG Encoder ---
class EEGEncoder(nn.Module):
    """Encodes EEG time-series into a latent vector using 1D CNN + Transformer."""
    def __init__(self, n_channels=64, seq_len=256, latent_dim=256):
        super().__init__()

        # 1D CNN for local temporal features
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
        )

        # Transformer for global temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8,
            dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(256, latent_dim)

    def forward(self, x):           # x: (B, 64, 256)
        x = self.cnn(x)             # (B, 256, 256)
        x = x.permute(0, 2, 1)     # (B, 256, 256) → transformer expects (B, seq, dim)
        x = self.transformer(x)     # (B, 256, 256)
        x = x.permute(0, 2, 1)     # (B, 256, 256)
        x = self.pool(x).squeeze(-1)  # (B, 256)
        return self.fc(x)           # (B, latent_dim)


# --- fMRI Encoder ---
class fMRIEncoder(nn.Module):
    """Encodes fMRI brain images into a latent vector using 2D CNN."""
    def __init__(self, latent_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),                              # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),                              # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),                              # 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)                       # (B, 256, 1, 1)
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):               # x: (B, 1, 64, 64)
        x = self.cnn(x).squeeze(-1).squeeze(-1)  # (B, 256)
        return self.fc(x)               # (B, latent_dim)


# --- MEG Encoder ---
class MEGEncoder(nn.Module):
    """Encodes MEG signals into a latent vector using 1D CNN."""
    def __init__(self, n_channels=306, seq_len=256, latent_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):               # x: (B, 306, 256)
        x = self.cnn(x).squeeze(-1)     # (B, 256)
        return self.fc(x)               # (B, latent_dim)


# --- Multimodal Fusion Encoder ---
class MultiModalEncoder(nn.Module):
    """
    Fuses EEG + fMRI + MEG into a single latent representation.
    Uses attention-based fusion to weight each modality dynamically.
    """
    def __init__(self, latent_dim=256):
        super().__init__()

        self.eeg_encoder  = EEGEncoder(latent_dim=latent_dim)
        self.fmri_encoder = fMRIEncoder(latent_dim=latent_dim)
        self.meg_encoder  = MEGEncoder(latent_dim=latent_dim)

        # Attention fusion: learns which modality to trust more
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, eeg, fmri, meg):
        # Encode each modality
        z_eeg  = self.eeg_encoder(eeg)    # (B, latent_dim)
        z_fmri = self.fmri_encoder(fmri)  # (B, latent_dim)
        z_meg  = self.meg_encoder(meg)    # (B, latent_dim)

        # Concatenate all modalities
        z_all = torch.cat([z_eeg, z_fmri, z_meg], dim=-1)  # (B, latent_dim*3)

        # Attention weights for each modality
        attn   = self.attention(z_all)              # (B, 3)
        z_eeg  = z_eeg  * attn[:, 0:1]
        z_fmri = z_fmri * attn[:, 1:2]
        z_meg  = z_meg  * attn[:, 2:3]

        # Fuse
        z_fused = torch.cat([z_eeg, z_fmri, z_meg], dim=-1)
        return self.fusion(z_fused)                 # (B, latent_dim)