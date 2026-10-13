"""
Gemini Agent for comparison testing
Uses Google Gemini API to generate responses for comparison with SLM
"""
import os
from typing import Dict, Optional
from agents.base_agent import BaseDialogueAgent
from dialogue_processor import DialogueProcessor


class GeminiAgent(BaseDialogueAgent):
    """Google Gemini-based dialogue agent for comparison testing"""
    
    def __init__(self, csv_path: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini agent
        
        Args:
            csv_path: Path to dialogue CSV
            model_name: Gemini model ID (gemini-1.5-flash, gemini-1.5-pro, etc.)
        """
        super().__init__("Gemini", model_name)
        self.csv_path = csv_path
        self.model_name = model_name
        self.api_key = None
        self.api_url = None
        self.conversation_history = []
        self.max_history = 5
        self.vector_store = None
        self.dialogue_processor = DialogueProcessor(csv_path)
        self.character_id = None
        self.character_name = None
        self.personality = None
    
    def initialize(self) -> bool:
        """Initialize the Gemini client"""
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Warning: GEMINI_API_KEY not found in environment")
                return False
            
            try:
                import requests
                self.api_key = api_key
                self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            except ImportError:
                print("Error: requests package not installed. Install with: pip install requests")
                return False
            
            print(f"Gemini agent initialized: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"Error initializing Gemini agent: {e}")
            import traceback
            traceback.print_exc()
            return False

    def use_character(self, character: Dict) -> None:
        """Point retrieval and the prompt at one discovered character."""
        from characters import personality_for
        from vector_store import open_vector_store

        self.character_id = character["id"]
        self.character_name = character["name"]
        self.personality = personality_for(character)
        self.dialogue_processor = DialogueProcessor(character["csv_path"])
        self.vector_store = open_vector_store(character["db_path"])
        self.reset_conversation()
    
    def build_prompt(
        self,
        user_input: str,
        context: str,
        emotion: Optional[str],
        include_examples: bool
    ) -> str:
        """Build context-aware prompt with personality and conversation history"""
        
        personality = self.personality or (
            "You are a character in a conversation. Stay in character. "
            "Keep responses to 1-2 sentences."
        )
        speaker = self.character_name or "Character"

        # Context-specific behavior
        context_map = {
            'investigation': "You're working on a case. Be methodical and observant.",
            'combat': "You're in danger. Stay focused and alert.",
            'emotional': "This is personal. Show some vulnerability.",
            'casual': "You're having a casual chat. Show your personality.",
            'greeting': "You're greeting someone. Be professional but friendly.",
            'moral_choice': "This involves a decision. Express your values.",
            'location': "You're commenting on a location. Share observations."
        }
        
        # Emotion-specific tone
        emotion_map = {
            'stern': "Be firm and serious.",
            'amused': "Show dry humor and sarcasm.",
            'concerned': "Express genuine concern.",
            'irritated': "Show impatience and frustration.",
            'somber': "Be reflective and thoughtful.",
            'puzzled': "Show curiosity and confusion.",
            'angry': "Show controlled anger.",
            'surprised': "Show genuine surprise."
        }
        
        # Build system message
        system = personality
        if context in context_map:
            system += f"\n\nCONTEXT: {context_map[context]}"
        if emotion and emotion in emotion_map:
            system += f"\n\nEMOTION: {emotion_map[emotion]}"
        
        # Add conversation history
        history_text = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]
            history_text = "\n\nRECENT CONVERSATION:\n"
            for exchange in recent_history:
                history_text += f"User: {exchange['user']}\n"
                history_text += f"{speaker}: {exchange['agent']}\n"
        
        # Add examples if requested and available
        examples_text = ""
        if include_examples and self.vector_store:
            try:
                vector_examples = self.vector_store.get_contextual_examples(
                    user_input=user_input,
                    context=context,
                    emotion=emotion,
                    n_results=3
                )
                if vector_examples:
                    examples_text = f"\n\nEXAMPLE NICK RESPONSES:\n{vector_examples}"
            except Exception as e:
                pass
        
        # Fallback to CSV examples if vector store fails
        if not examples_text and include_examples:
            try:
                csv_examples = self.dialogue_processor.get_random_examples(2)
                if csv_examples:
                    example_list = [ex.get('RESPONSE TEXT', '').strip() for ex in csv_examples if ex.get('RESPONSE TEXT')]
                    if example_list:
                        examples_text = "\n\nEXAMPLE NICK RESPONSES:\n" + "\n".join([f'"{ex}"' for ex in example_list])
            except Exception as e:
                pass
        
        # Combine into full prompt
        full_prompt = f"{system}{history_text}{examples_text}\n\nUser: {user_input}\n\nNick:"
        
        return full_prompt
    
    def generate_response(
        self,
        user_input: str,
        context: str = "casual",
        emotion: Optional[str] = None,
        include_examples: bool = True
    ) -> Dict:
        """Generate response using Google Gemini model"""
        
        if not hasattr(self, 'api_key') or not self.api_key:
            raise Exception("Agent not initialized. Call initialize() first.")
        if not self.character_id:
            return {
                'response': "No character is loaded. Add a dialogue CSV and build its vector database.",
                'error': 'no_character',
                'agent': self.name,
                'model': self.model_name,
            }
        
        try:
            # Build prompt
            prompt = self.build_prompt(user_input, context, emotion, include_examples)
            
            # Generate response via API
            import requests
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.9,
                    "maxOutputTokens": 80
                }
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            nick_response = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Clean response (remove any meta-text)
            meta_patterns = [
                'Context:', 'Emotion:', 'Recent conversation:', 
                'Sure,', 'Certainly', "Here's", 'edited version', 
                'longer response', 'Example responses:'
            ]
            for pattern in meta_patterns:
                if pattern in nick_response:
                    nick_response = nick_response.split(pattern)[0].strip()
            
            # Stop at dialogue markers
            for marker in ['User:', 'user:', 'Nick:', 'nick:']:
                if marker in nick_response:
                    nick_response = nick_response.split(marker)[0].strip()
            
            # Clean newlines
            nick_response = nick_response.replace('\n\n', ' ').replace('\n', ' ').strip()
            
            # Limit to 2 sentences max
            sentences = [s.strip() for s in nick_response.split('. ') if s.strip()]
            if len(sentences) > 2:
                nick_response = '. '.join(sentences[:2])
                if not nick_response.endswith('.'):
                    nick_response += '.'
            elif sentences and not nick_response.endswith('.') and not nick_response.endswith('?') and not nick_response.endswith('!'):
                nick_response += '.'
            
            # Remove quotes and brackets
            nick_response = nick_response.strip('"\'\'[]()').strip()
            
            # Validate response quality
            if len(nick_response.strip()) < 5 or not any(c.isalpha() for c in nick_response):
                print(f"Warning: Generated response invalid: '{nick_response}'")
                fallbacks = {
                    'investigation': "Let me think about this.",
                    'combat': "Stay sharp.",
                    'emotional': "It's complicated.",
                    'greeting': "Hello there.",
                    'casual': "What can I do for you?"
                }
                nick_response = fallbacks.get(context, "What can I do for you?")
            
            # Add to history
            self.add_to_history(user_input, nick_response, context, emotion)
            
            return {
                'response': nick_response,
                'context': context,
                'emotion': emotion,
                'agent': self.name,
                'model': self.model_name,
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to CSV
            fallback = self.dialogue_processor.get_random_examples(1)
            fallback_text = fallback[0].get('RESPONSE TEXT', "Something's not right here...") if fallback else "Something's not right here..."
            
            return {
                'response': fallback_text,
                'error': str(e),
                'fallback': True,
                'agent': self.name,
            }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_info(self) -> Dict:
        """Get agent information"""
        return {
            'name': self.name,
            'model': self.model_name,
            'ready': self.api_key is not None,
            'type': 'gemini'
        }

