import torch.nn.functional as F
import torch
import torch.nn as nn
from encoder import MultiModalEncoder
from decoder import MentalStateDecoder, EEGSignalDecoder, fMRIDecoder


class TetherBCI(nn.Module):
    """
    TetherBCI: Open-Source Brain-Computer Interface Framework
    
    A modular framework that:
    1. Encodes multimodal brain signals (EEG + fMRI + MEG)
    2. Fuses them into a unified latent representation
    3. Decodes into mental state predictions or reconstructed signals
    """
    def __init__(self, latent_dim=256, n_classes=4):
        super().__init__()

        # --- Encoders ---
        self.encoder = MultiModalEncoder(latent_dim=latent_dim)

        # --- Decoders ---
        self.mental_state_decoder = MentalStateDecoder(latent_dim, n_classes)
        self.eeg_decoder          = EEGSignalDecoder(latent_dim)
        self.fmri_decoder         = fMRIDecoder(latent_dim)

        # --- Layer Norm for stable training ---
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, eeg, fmri, meg, mode="classify"):
        """
        Forward pass with multiple modes:
        - classify:     predict mental state
        - reconstruct:  reconstruct EEG + fMRI signals
        - full:         both classify + reconstruct
        """
        # Encode all modalities
        z = self.encoder(eeg, fmri, meg)  # (B, latent_dim)
        z = self.layer_norm(z)

        if mode == "classify":
            return self.mental_state_decoder(z)

        elif mode == "reconstruct":
            return {
                "eeg_recon":  self.eeg_decoder(z),
                "fmri_recon": self.fmri_decoder(z)
            }

        elif mode == "full":
            return {
                "logits":     self.mental_state_decoder(z),
                "eeg_recon":  self.eeg_decoder(z),
                "fmri_recon": self.fmri_decoder(z),
                "latent":     z
            }

    def encode(self, eeg, fmri, meg):
        """Extract latent representation only — used for downstream tasks."""
        z = self.encoder(eeg, fmri, meg)
        return self.layer_norm(z)

    def count_parameters(self):
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return total, trainable


class BCILoss(nn.Module):
    """
    Combined loss for TetherBCI:
    - Classification loss (CrossEntropy)
    - EEG reconstruction loss (MSE)
    - fMRI reconstruction loss (MSE)
    """
    def __init__(self, cls_weight=1.0, recon_weight=0.5):
        super().__init__()
        self.cls_weight   = cls_weight
        self.recon_weight = recon_weight
        self.cls_loss     = nn.CrossEntropyLoss()
        self.recon_loss   = nn.MSELoss()

    def forward(self, outputs, labels, eeg_target, fmri_target):
        # Classification loss
        cls   = self.cls_loss(outputs["logits"], labels)

        # Reconstruction losses
        # Slice EEG recon to match target size
        eeg_recon   = torch.nn.functional.interpolate(outputs["eeg_recon"], size=eeg_target.shape[-1])
        recon_eeg   = self.recon_loss(eeg_recon, eeg_target)
        recon_fmri  = self.recon_loss(outputs["fmri_recon"], fmri_target)

        total = self.cls_weight * cls + self.recon_weight * (recon_eeg + recon_fmri)
        return total, cls, recon_eeg, recon_fmri