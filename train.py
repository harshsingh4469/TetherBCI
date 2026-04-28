import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from framework import TetherBCI, BCILoss
from data_loader import MultiModalBCIDataset, generate_multimodal_data

# --- Config ---
DEVICE     = "cpu"
EPOCHS     = 30
BATCH      = 8
LR         = 1e-4
LATENT_DIM = 256
N_CLASSES  = 4

def train():
    # Generate data if not exists
    if not os.path.exists("data/eeg"):
        generate_multimodal_data()

    # Dataset
    dataset    = MultiModalBCIDataset("data/eeg", "data/fmri", "data/meg")
    val_size   = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    # Model
    model     = TetherBCI(latent_dim=LATENT_DIM, n_classes=N_CLASSES).to(DEVICE)
    criterion = BCILoss(cls_weight=1.0, recon_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("=" * 55)
    print("TetherBCI Training")
    print("=" * 55)
    model.count_parameters()
    print(f"\nTraining on {train_size} samples, validating on {val_size}\n")

    best_val_acc = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_losses, train_correct = [], 0

        for eeg, fmri, meg, labels in train_loader:
            eeg, fmri, meg = eeg.to(DEVICE), fmri.to(DEVICE), meg.to(DEVICE)
            labels         = labels.to(DEVICE)

            outputs = model(eeg, fmri, meg, mode="full")
            loss, cls, recon_eeg, recon_fmri = criterion(
                outputs, labels, eeg, fmri
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds          = outputs["logits"].argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        train_acc = train_correct / train_size

        # --- Validation ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for eeg, fmri, meg, labels in val_loader:
                eeg, fmri, meg = eeg.to(DEVICE), fmri.to(DEVICE), meg.to(DEVICE)
                labels         = labels.to(DEVICE)
                logits         = model(eeg, fmri, meg, mode="classify")
                preds          = logits.argmax(dim=1)
                val_correct   += (preds == labels).sum().item()

        val_acc = val_correct / val_size

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {np.mean(train_losses):.4f} | "
              f"Train Acc: {train_acc*100:.1f}% | "
              f"Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/tetherbci.pt")
            print(f"  ✅ Best model saved! Val Acc: {val_acc*100:.1f}%")

        scheduler.step()

    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc*100:.1f}%")
    print("Model saved as models/tetherbci.pt")

if __name__ == "__main__":
    train()