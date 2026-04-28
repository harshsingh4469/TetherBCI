import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from framework import TetherBCI
from data_loader import MultiModalBCIDataset

# --- Load model ---
model = TetherBCI()
model.load_state_dict(torch.load("models/tetherbci.pt"))
model.eval()

# --- Load dataset ---
dataset = MultiModalBCIDataset("data/eeg", "data/fmri", "data/meg")
labels_map = {0: "Rest", 1: "Motor", 2: "Visual", 3: "Cognitive"}

# --- Figure 1: Multimodal Input Visualization ---
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("TetherBCI: Multimodal Brain Signals", fontsize=16, fontweight='bold')

for i in range(4):
    eeg, fmri, meg, label = dataset[i]

    # EEG plot
    axes[0, i].plot(eeg[0].numpy(), color='#3498db', linewidth=0.8)
    axes[0, i].plot(eeg[1].numpy(), color='#e74c3c', linewidth=0.8, alpha=0.7)
    axes[0, i].set_title(f"{labels_map[label]} — EEG")
    axes[0, i].set_xlabel("Timepoints")
    axes[0, i].set_ylabel("Amplitude")

    # fMRI plot
    axes[1, i].imshow(fmri[0].numpy(), cmap='hot')
    axes[1, i].set_title(f"{labels_map[label]} — fMRI")
    axes[1, i].axis('off')

    # MEG plot
    axes[2, i].plot(meg[0].numpy(), color='#2ecc71', linewidth=0.8)
    axes[2, i].plot(meg[1].numpy(), color='#9b59b6', linewidth=0.8, alpha=0.7)
    axes[2, i].set_title(f"{labels_map[label]} — MEG")
    axes[2, i].set_xlabel("Timepoints")
    axes[2, i].set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig("results/multimodal_signals.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/multimodal_signals.png")

# --- Figure 2: Mental State Classification ---
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("TetherBCI: Mental State Classification", fontsize=16, fontweight='bold')

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for i in range(40):
        eeg, fmri, meg, label = dataset[i]
        logits = model(eeg.unsqueeze(0), fmri.unsqueeze(0), meg.unsqueeze(0), mode="classify")
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
        pred   = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(probs)

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
cm  = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_map.values(),
            yticklabels=labels_map.values(), ax=axes2[0])
axes2[0].set_title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
axes2[0].set_ylabel("True Label")
axes2[0].set_xlabel("Predicted Label")

# Average probabilities per class
avg_probs = np.array(all_probs).mean(axis=0)
colors    = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
axes2[1].bar(labels_map.values(), avg_probs, color=colors, width=0.5)
axes2[1].set_title("Average Prediction Confidence per Class")
axes2[1].set_ylabel("Probability")
axes2[1].set_ylim(0, 1)
for i, v in enumerate(avg_probs):
    axes2[1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("results/classification_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/classification_results.png")

# --- Figure 3: Latent Space Visualization ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
fig3.suptitle("TetherBCI: Latent Space (Brain Representations)", fontsize=14, fontweight='bold')

from sklearn.decomposition import PCA
latents, label_list = [], []
with torch.no_grad():
    for i in range(len(dataset)):
        eeg, fmri, meg, label = dataset[i]
        z = model.encode(eeg.unsqueeze(0), fmri.unsqueeze(0), meg.unsqueeze(0))
        latents.append(z.squeeze().numpy())
        label_list.append(label)

latents = np.array(latents)
pca     = PCA(n_components=2)
reduced = pca.fit_transform(latents)

colors  = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
for cls in range(4):
    idx = [i for i, l in enumerate(label_list) if l == cls]
    ax3.scatter(reduced[idx, 0], reduced[idx, 1],
                c=colors[cls], label=labels_map[cls],
                alpha=0.7, s=80, edgecolors='white', linewidth=0.5)

ax3.set_title("PCA of Latent Brain Representations")
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.legend()
plt.tight_layout()
plt.savefig("results/latent_space.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/latent_space.png")
print("\nAll visualizations saved in results/")