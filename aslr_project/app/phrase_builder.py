import json
from pathlib import Path
from typing import List, Dict, Any, Optional

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "data" / "metadata" / "phrase_templates.json"

class PhraseBuilder:
    def __init__(self, templates_file: Optional[Path] = None):
        path = templates_file or TEMPLATES_PATH
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                self.templates = json.load(f)
        else:
            self.templates = {}
            
    def build_phrase(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Maps a sequence of recognized isolated ASL tokens to a canonical phrase
        with English, Tamil, and Hindi translations.
        """
        if not tokens:
            return {
                "matched": False,
                "approximate": False,
                "en": "",
                "ta": "",
                "hi": "",
                "tokens": []
            }
            
        token_set = set(t.lower().strip() for t in tokens)
        
        # Check for exact subset match in templates
        best_match = None
        best_overlap = 0
        
        for key, tmpl in self.templates.items():
            tmpl_set = set(t.lower().strip() for t in tmpl["tokens"])
            if tmpl_set.issubset(token_set):
                overlap = len(tmpl_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = tmpl
                    
        if best_match is not None:
            return {
                "matched": True,
                "approximate": False,
                "en": best_match["en"],
                "ta": best_match["ta"],
                "hi": best_match["hi"],
                "tokens": best_match["tokens"]
            }
            
        # Fallback approximate translation
        en_fallback = " ".join(tokens).capitalize() + "."
        return {
            "matched": False,
            "approximate": True,
            "en": en_fallback,
            "ta": "தோராயமான மொழிபெயர்ப்பு: " + " ".join(tokens),
            "hi": "अनुमानित अनुवाद: " + " ".join(tokens),
            "tokens": tokens
        }
