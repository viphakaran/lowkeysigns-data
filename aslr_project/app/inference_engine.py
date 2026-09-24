import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bigru_classifier.pth"

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, gru_output):
        scores = self.attn(gru_output)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(gru_output * weights, dim=1)
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
        out, _ = self.gru(x)
        context, _ = self.attn(out)
        context = self.dropout(context)
        logits = self.classifier(context)
        return logits

class ASLInferenceEngine:
    def __init__(self, model_checkpoint: Path = MODEL_PATH):
        self.device = torch.device("cpu")
        ckpt = torch.load(str(model_checkpoint), map_location=self.device)
        
        self.target_glosses = ckpt["target_glosses"]
        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.model = BiGRUSignClassifier(
            input_dim=ckpt.get("input_dim", 268),
            hidden_dim=ckpt.get("hidden_dim", 64),
            num_layers=2,
            num_classes=ckpt.get("num_classes", len(self.target_glosses)),
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
    def predict(self, sequence: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Accepts a (T, 268) or (60, 268) sequence.
        Returns: (top1_label, top1_confidence, top3_predictions)
        """
        T, D = sequence.shape
        padded = np.zeros((60, 268), dtype=np.float32)
        if T < 60:
            padded[:T, :D] = sequence
        else:
            padded[:, :D] = sequence[:60, :D]
            
        tensor_x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            
        top_indices = np.argsort(probs)[::-1][:3]
        
        top3 = []
        for idx in top_indices:
            top3.append({
                "label": self.idx_to_class[int(idx)],
                "confidence": float(probs[idx])
            })
            
        top1_label = top3[0]["label"]
        top1_conf = top3[0]["confidence"]
        return top1_label, top1_conf, top3
