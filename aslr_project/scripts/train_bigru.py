import os
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

TARGET_GLOSSES = [
    "help", "wait", "money", "form", "pain",
    "doctor", "yes", "no", "thank you", "sign",
    "more", "problem", "emergency", "where",
    "name", "appointment", "sick", "please",
    "here", "now"
]

class_to_idx = {g: i for i, g in enumerate(TARGET_GLOSSES)}
idx_to_class = {i: g for i, g in enumerate(TARGET_GLOSSES)}

LABEL_MAP_PATH = Path("data/metadata/label_mapping.json")
LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
with LABEL_MAP_PATH.open("w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "bigru_classifier.pth"
CONF_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
EVAL_REPORT_PATH = MODEL_DIR / "evaluation_report.json"

FIXED_FRAMES = 60
FEATURE_DIM = 268

def load_split_data(split_csv_path):
    with open(split_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    X_list = []
    y_list = []
    meta_list = []
    
    for row in rows:
        fpath = Path(row["path"])
        if not fpath.exists():
            continue
        label = row["label"]
        if label not in class_to_idx:
            continue
            
        seq = np.load(str(fpath))
        T, D = seq.shape
        padded = np.zeros((FIXED_FRAMES, FEATURE_DIM), dtype=np.float32)
        if T < FIXED_FRAMES:
            padded[:T, :D] = seq
        else:
            padded[:, :D] = seq[:FIXED_FRAMES, :D]
            
        X_list.append(padded)
        y_list.append(class_to_idx[label])
        meta_list.append(row)
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), meta_list

print("Loading dataset splits...")
X_train, y_train, train_meta = load_split_data("data/splits/train.csv")
X_val, y_val, val_meta = load_split_data("data/splits/val.csv")
X_test, y_test, test_meta = load_split_data("data/splits/test.csv")

print(f"X_train shape: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test shape:  {X_test.shape}, y_test: {y_test.shape}")

class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ASLDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(ASLDataset(X_val, y_val), batch_size=16, shuffle=False)
test_loader = DataLoader(ASLDataset(X_test, y_test), batch_size=16, shuffle=False)

# Attention Layer for temporal weighting
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, gru_output):
        # gru_output: (batch, seq_len, hidden_dim)
        scores = self.attn(gru_output)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(gru_output * weights, dim=1)  # (batch, hidden_dim)
        return context, weights

class BiGRUSignClassifier(nn.Module):
    def __init__(self, input_dim=268, hidden_dim=64, num_layers=2, num_classes=20, dropout=0.35):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = TemporalAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 60, 268)
        out, _ = self.gru(x)  # (batch, 60, hidden_dim * 2)
        context, _ = self.attn(out)  # (batch, hidden_dim * 2)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits

device = torch.device("cpu")
model = BiGRUSignClassifier(input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, num_classes=len(TARGET_GLOSSES), dropout=0.35).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Bi-GRU + Attention Model initialized: {total_params:,} parameters.")

# Class weights
class_counts = np.bincount(y_train, minlength=len(TARGET_GLOSSES))
weights = len(y_train) / (len(TARGET_GLOSSES) * class_counts.astype(np.float32))
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

EPOCHS = 60
best_val_loss = float("inf")
best_model_weights = None
best_epoch = 0

print("\nStarting training loop...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        train_loss += loss.item() * bx.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == by).sum().item()
        train_total += bx.size(0)
        
    scheduler.step()
    
    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            val_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == by).sum().item()
            val_total += bx.size(0)
            
    train_loss /= train_total
    train_acc = train_correct / train_total
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_weights = model.state_dict().copy()
        
    if epoch % 5 == 0 or epoch == 1 or epoch == best_epoch:
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

print(f"\nTraining completed! Restoring best weights from epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
model.load_state_dict(best_model_weights)

# Save checkpoint
torch.save({
    "epoch": best_epoch,
    "model_state_dict": model.state_dict(),
    "input_dim": FEATURE_DIM,
    "hidden_dim": 64,
    "num_classes": len(TARGET_GLOSSES),
    "target_glosses": TARGET_GLOSSES,
    "class_to_idx": class_to_idx,
    "best_val_loss": best_val_loss
}, str(MODEL_SAVE_PATH))

print(f"Model saved to {MODEL_SAVE_PATH}")

# Evaluation on Test Split (Unseen real samples)
model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for bx, by in test_loader:
        logits = model(bx.to(device))
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(by.numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_acc = (test_preds == test_targets).mean()

print(f"\n==========================================")
print(f"FINAL UNSEEN TEST ACCURACY: {test_acc * 100:.2f}%")
print(f"==========================================")

report = classification_report(
    test_targets, test_preds,
    labels=list(range(len(TARGET_GLOSSES))),
    target_names=TARGET_GLOSSES,
    digits=3,
    output_dict=True,
    zero_division=0
)

# Plot confusion matrix
cm = confusion_matrix(test_targets, test_preds, labels=list(range(len(TARGET_GLOSSES))))
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=TARGET_GLOSSES,
    yticklabels=TARGET_GLOSSES
)
plt.title(f"Bi-GRU Test Confusion Matrix (Accuracy: {test_acc*100:.1f}%)", fontsize=14, pad=15)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(str(CONF_MATRIX_PATH), dpi=150)
plt.close()

print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

eval_data = {
    "model": "Bi-GRU + Temporal Attention",
    "best_epoch": best_epoch,
    "best_val_loss": float(best_val_loss),
    "test_accuracy": float(test_acc),
    "pain_metrics": report.get("pain", {}),
    "emergency_metrics": report.get("emergency", {}),
    "classification_report": report
}

with EVAL_REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(eval_data, f, indent=2)

print(f"Saved evaluation report to {EVAL_REPORT_PATH}")
