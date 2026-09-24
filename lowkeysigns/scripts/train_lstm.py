import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data and labels
DATASET_PATH = "data/train_ready_dataset.npz"
LABEL_MAPPING_PATH = "data/label_mapping.json"
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "lstm_classifier.pth"
CONF_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"

with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)

# Sort labels by integer key
num_classes = len(label_mapping)
class_names = [label_mapping[str(i)] for i in range(num_classes)]

raw_data = np.load(DATASET_PATH)
X = raw_data["X"]  # (135, 60, 254)
y = raw_data["y"]  # (135,)

print(f"Loaded X: {X.shape}, y: {y.shape}")

# Requirement 1: Stratified train/val split (80/20)
# Stratify ensures each class (including 4-sample classes) gets validation samples
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")

# Verify split distribution
print("\nClass split distribution:")
print(f"{'Class Index':<12} | {'Class Name':<15} | {'Train Count':<12} | {'Val Count':<10}")
print("-" * 55)
for i in range(num_classes):
    n_train = int(np.sum(y_train == i))
    n_val = int(np.sum(y_val == i))
    print(f"{i:<12} | {class_names[i]:<15} | {n_train:<12} | {n_val:<10}")

# PyTorch Dataset
class ASLLandmarkDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ASLLandmarkDataset(X_train, y_train)
val_dataset = ASLLandmarkDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Requirement 2: Small LSTM Classifier in PyTorch
class SignLSTMClassifier(nn.Module):
    def __init__(self, input_size=254, hidden_size=64, num_layers=1, num_classes=20, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len=60, input_size=254)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last hidden state hn[-1]: (batch_size, hidden_size)
        out = self.dropout(hn[-1])
        logits = self.fc(out)
        return logits

device = torch.device("cpu")
model = SignLSTMClassifier(input_size=254, hidden_size=64, num_layers=1, num_classes=num_classes, dropout=0.4).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel initialized: {total_params} trainable parameters.")

# Requirement 4: Class-weighted loss to assist thin classes (pain, emergency)
class_counts = np.bincount(y_train, minlength=num_classes)
# Inverse frequency weighting
weights = len(y_train) / (num_classes * class_counts.astype(np.float32))
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Requirement 5: Train with early stopping monitoring validation loss
max_epochs = 80
patience = 15
best_val_loss = float("inf")
patience_counter = 0
best_epoch = 0
best_model_state = None

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": []
}

print("\nStarting training loop...")
for epoch in range(1, max_epochs + 1):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
        
    train_loss = running_loss / total
    train_acc = correct / total
    
    # Validation phase
    model.eval()
    val_loss_running = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            val_loss_running += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)
            
    val_loss = val_loss_running / val_total
    val_acc = val_correct / val_total
    
    scheduler.step(val_loss)
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    
    if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
        print(f"Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
        
    # Check early stopping & best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with val loss {best_val_loss:.4f}.")
            break

# Load best model weights
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save the trained model checkpoint (Requirement 7)
torch.save({
    "epoch": best_epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "input_size": 254,
    "hidden_size": 64,
    "num_classes": num_classes,
    "label_mapping": label_mapping,
    "best_val_loss": best_val_loss
}, str(MODEL_SAVE_PATH))

print(f"\nModel checkpoint saved to {MODEL_SAVE_PATH}")

# Requirement 6: Validation Evaluation & Confusion Matrix
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        outputs = model(batch_X.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

val_final_acc = (all_preds == all_targets).mean()
print(f"\nFinal Best Model Validation Accuracy: {val_final_acc * 100:.2f}%")

# Classification Report
report = classification_report(
    all_targets, all_preds,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=3,
    output_dict=True,
    zero_division=0
)

# Confusion Matrix (Requirement 8)
cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title(f"Confusion Matrix (Val Accuracy: {val_final_acc*100:.1f}%)", fontsize=14, pad=15)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(str(CONF_MATRIX_PATH), dpi=150)
plt.close()

print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

# Save detailed evaluation report as JSON
eval_summary = {
    "best_epoch": best_epoch,
    "best_val_loss": float(best_val_loss),
    "final_val_accuracy": float(val_final_acc),
    "pain_metrics": report.get("pain", {}),
    "emergency_metrics": report.get("emergency", {}),
    "classification_report": report
}

with open("model/evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

print("Saved evaluation report to model/evaluation_report.json")
