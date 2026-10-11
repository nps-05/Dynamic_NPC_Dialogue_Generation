"""
Dialogue Data Processor
Parses Nick Valentine's dialogue CSV and provides utilities for context retrieval
"""

import pandas as pd
import random
from typing import List, Dict, Optional


KNOWN_CONTEXTS = (
    "investigation",
    "combat",
    "emotional",
    "casual",
    "greeting",
    "moral_choice",
    "location",
)
KNOWN_EMOTIONS = (
    "neutral",
    "amused",
    "stern",
    "concerned",
    "irritated",
    "somber",
    "puzzled",
    "angry",
    "surprised",
)

# First match wins. Combat and investigation are checked before broader labels.
_CONTEXT_KEYWORDS = (
    ("combat", ("combat", "weapon", "fight", "danger", "kill", "threat", "gun", "ambush")),
    ("investigation", ("investigation", "case", "clue", "evidence", "murder", "witness", "suspect", "alibi", "crime")),
    ("emotional", ("emotional", "lonely", "grief", "personal")),
    ("greeting", ("greeting", "greet", "hello", "welcome")),
    ("moral_choice", ("moral", "conscience", "right thing")),
    ("location", ("location", "alley", "docks", "district")),
)
_EMOTION_KEYWORDS = (
    ("amused", ("happy", "amused", "wry", "sarcastic")),
    ("stern", ("stern", "serious", "firm")),
    ("concerned", ("concerned", "worried", "afraid", "anxious")),
    ("irritated", ("irritated", "annoyed", "impatient")),
    ("somber", ("somber", "sad", "grief", "mourn")),
    ("puzzled", ("puzzled", "confused", "curious")),
    ("angry", ("angry", "furious")),
    ("surprised", ("surprised", "shocked", "startled")),
    ("neutral", ("neutral", "calm")),
)


def _cell(row: Dict, *keys: str) -> str:
    for key in keys:
        if key not in row or not pd.notna(row.get(key)):
            continue
        value = str(row.get(key)).strip()
        if value and value.lower() != "nan":
            return value
    return ""


def infer_retrieval_labels(row: Dict) -> Dict[str, str]:
    """Label a dialogue row for vector search.

    Explicit CONTEXT and EMOTION columns win. Otherwise keywords are taken from
    notes, category, type, scene, and the line itself. Unlabeled rows stay
    casual/neutral instead of being dropped.
    """
    explicit_context = _cell(row, "CONTEXT", "context").lower()
    explicit_emotion = _cell(row, "EMOTION", "emotion").lower()
    blob = " ".join(
        [
            _cell(row, "SCRIPT NOTES"),
            _cell(row, "CATEGORY"),
            _cell(row, "TYPE"),
            _cell(row, "SCENE"),
            _cell(row, "RESPONSE TEXT"),
        ]
    ).lower()

    context = explicit_context if explicit_context in KNOWN_CONTEXTS else ""
    emotion = explicit_emotion if explicit_emotion in KNOWN_EMOTIONS else ""

    if not context:
        for label, words in _CONTEXT_KEYWORDS:
            if any(word in blob for word in words):
                context = label
                break
    if not emotion:
        for label, words in _EMOTION_KEYWORDS:
            if any(word in blob for word in words):
                emotion = label
                break

    return {
        "context": context or "casual",
        "emotion": emotion or "neutral",
    }


class DialogueProcessor:
    def __init__(self, csv_path: str = "data/nick_valentine_dialogue.csv"):
        self.csv_path = csv_path
        self.dialogues = []
        self.load_dialogues()
    
    def load_dialogues(self):
        """Load and parse the CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Convert to list of dictionaries
            self.dialogues = df.to_dict('records')
            
            # Filter out empty responses
            self.dialogues = [
                d for d in self.dialogues 
                if pd.notna(d.get('RESPONSE TEXT')) and str(d.get('RESPONSE TEXT')).strip()
            ]
            
            print(f"✓ Loaded {len(self.dialogues)} dialogue entries")
        except Exception as e:
            print(f"Error loading dialogue CSV: {e}")
            self.dialogues = []
    
    def get_random_examples(self, count: int = 5) -> List[Dict]:
        """Get random dialogue examples"""
        if not self.dialogues:
            return []
        return random.sample(self.dialogues, min(count, len(self.dialogues)))
    
    def get_examples_by_emotion(self, emotion: str, count: int = 3) -> List[Dict]:
        """Get examples by emotion/tone"""
        filtered = [
            d for d in self.dialogues
            if pd.notna(d.get('SCRIPT NOTES')) and 
            emotion.lower() in str(d.get('SCRIPT NOTES')).lower()
        ]
        return filtered[:count]
    
    def get_examples_by_category(self, category: str, count: int = 5) -> List[Dict]:
        """Get examples by category"""
        filtered = [
            d for d in self.dialogues
            if d.get('CATEGORY') == category
        ]
        return filtered[:count]
    
    def get_examples_by_context(self, context: str, count: int = 5) -> List[Dict]:
        """Get examples by context/scene type"""
        context_keywords = {
            'investigation': ['case', 'clue', 'evidence', 'murder', 'investigating', 'detective'],
            'combat': ['weapon', 'fight', 'danger', 'kill', 'threat'],
            'casual': ['hello', 'greeting', 'idle', 'miscellaneous'],
            'emotional': ['sad', 'angry', 'depressed', 'concerned', 'somber'],
            'location': ['diamond city', 'commonwealth', 'institute', 'place'],
        }
        
        keywords = context_keywords.get(context, [])
        filtered = []
        
        for d in self.dialogues:
            text = ' '.join([
                str(d.get('RESPONSE TEXT', '')),
                str(d.get('SCRIPT NOTES', '')),
                str(d.get('SCENE', ''))
            ]).lower()
            
            if any(keyword in text for keyword in keywords):
                filtered.append(d)
        
        return filtered[:count]
    
    def search_dialogues(self, keyword: str, count: int = 5) -> List[Dict]:
        """Search dialogues by keyword"""
        search_term = keyword.lower()
        filtered = []
        
        for d in self.dialogues:
            text = ' '.join([
                str(d.get('RESPONSE TEXT', '')),
                str(d.get('SCRIPT NOTES', ''))
            ]).lower()
            
            if search_term in text:
                filtered.append(d)
        
        return filtered[:count]
    
    def format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """Format examples for prompt context"""
        if not examples:
            return ''
        
        formatted = []
        for ex in examples:
            text = str(ex.get('RESPONSE TEXT', '')).strip()
            notes = str(ex.get('SCRIPT NOTES', '')).strip()
            
            line = f'Nick: "{text}"'
            if notes and notes != 'nan':
                line += f' [{notes}]'
            formatted.append(line)
        
        return '\n'.join(formatted)
    
    def get_contextual_examples(
        self, 
        context: Optional[str] = None, 
        emotion: Optional[str] = None, 
        count: int = 5
    ) -> str:
        """Get contextual examples with formatting"""
        examples = []
        
        if emotion:
            examples = self.get_examples_by_emotion(emotion, count)
        elif context:
            examples = self.get_examples_by_context(context, count)
        
        # Fallback to random if no specific matches
        if len(examples) < count:
            needed = count - len(examples)
            examples.extend(self.get_random_examples(needed))
        
        return self.format_examples_for_prompt(examples)
    
    def get_stats(self) -> Dict:
        """Get dialogue statistics"""
        if not self.dialogues:
            return {}
        
        df = pd.DataFrame(self.dialogues)
        
        return {
            'total': len(self.dialogues),
            'categories': df['CATEGORY'].unique().tolist() if 'CATEGORY' in df else [],
            'types': df['TYPE'].unique().tolist() if 'TYPE' in df else [],
            'scenes': df['SCENE'].nunique() if 'SCENE' in df else 0,
        }
