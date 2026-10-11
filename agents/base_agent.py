"""
Base Agent Interface
All dialogue generation agents inherit from this
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseDialogueAgent(ABC):
    """Base class for all dialogue generation agents"""
    
    def __init__(self, name: str, model_name: str):
        self.name = name
        self.model_name = model_name
        self.conversation_history: List[Dict] = []
        self.max_history = 10
    
    @abstractmethod
    def generate_response(
        self, 
        user_input: str, 
        context: str = "casual",
        emotion: Optional[str] = None,
        include_examples: bool = True
    ) -> Dict:
        """Generate a response from the agent"""
        pass
    
    @abstractmethod
    def initialize(self):
        """Initialize the agent (load model, connect to API, etc.)"""
        pass
    
    def add_to_history(self, user_input: str, response: str, context: str, emotion: Optional[str]):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            'user': user_input,
            'agent': response,
            'context': context,
            'emotion': emotion,
        })
        
        # Keep only last N exchanges
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        print(f"✓ {self.name} conversation history reset")
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_info(self) -> Dict:
        """Get agent information"""
        return {
            'name': self.name,
            'model': self.model_name,
            'history_length': len(self.conversation_history),
        }
