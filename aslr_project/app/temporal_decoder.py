from collections import Counter
from typing import List, Tuple, Optional

class TemporalDecoder:
    def __init__(
        self,
        window_size: int = 8,
        min_agreement: int = 5,
        confidence_threshold: float = 0.75,
        cooldown_frames: int = 12
    ):
        self.window_size = window_size
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        
        self.history: List[Tuple[str, float]] = []
        self.last_committed_label: Optional[str] = None
        self.cooldown_counter: int = 0
        
    def reset(self):
        self.history.clear()
        self.last_committed_label = None
        self.cooldown_counter = 0

    def step(self, label: str, confidence: float) -> Optional[str]:
        """
        Processes a single-frame prediction (label, confidence).
        Returns the committed label if a stable sign is finalized, else None.
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None
            
        self.history.append((label, confidence))
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # Filter predictions by confidence threshold
        confident_preds = [
            lbl for lbl, conf in self.history 
            if conf >= self.confidence_threshold
        ]
        
        if len(confident_preds) < self.min_agreement:
            return None
            
        # Majority vote
        counts = Counter(confident_preds)
        most_common_label, count = counts.most_common(1)[0]
        
        if count >= self.min_agreement:
            if most_common_label != self.last_committed_label:
                self.last_committed_label = most_common_label
                self.cooldown_counter = self.cooldown_frames
                self.history.clear()
                return most_common_label
                
        return None
