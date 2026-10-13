"""
HuggingFace Agent for dialogue generation with vector DB integration
"""
import os
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from agents.base_agent import BaseDialogueAgent
from dialogue_processor import DialogueProcessor


class HuggingFaceAgent(BaseDialogueAgent):
    """HuggingFace-based dialogue agent using transformer models + vector search"""
    
    def __init__(self, csv_path: str, model_name: str = None):
        """
        Initialize HuggingFace agent
        
        Args:
            csv_path: Path to dialogue CSV
            model_name: HuggingFace model ID
        """
        from model_defaults import DEFAULT_HF_MODEL

        super().__init__("HuggingFace", model_name or DEFAULT_HF_MODEL)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history = []
        self.max_history = 5
        self.vector_store = None
        self.dialogue_processor = DialogueProcessor(csv_path)
        self.character_id = None
        self.character_name = None
        self.personality = None
    
    def initialize(self) -> bool:
        """Initialize the HuggingFace model"""
        try:
            # Check for HF token
            hf_token = os.environ.get('HF_TOKEN')
            if not hf_token:
                print("Warning: HF_TOKEN not found in environment")
            
            print(f"Loading HuggingFace model: {self.model_name}")
            print(f"   Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
                padding_side='left'
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Use memory-efficient loading to avoid paging file issues
            model_kwargs = {
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True,  # Reduces memory usage during loading
            }
            if hf_token:
                model_kwargs['token'] = hf_token
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except OSError as e:
                if "paging file" in str(e) or "1455" in str(e):
                    print("Error: Not enough memory to load model. Trying with device_map='cpu'...")
                    # Try loading directly to CPU first, then move to GPU if needed
                    model_kwargs['device_map'] = 'cpu'
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                    if self.device == "cuda":
                        print("Moving model to GPU...")
                        self.model = self.model.to(self.device)
                else:
                    raise
            
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Create text generation pipeline optimized for speed
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=1  # Single generation for faster response
            )
            
            print(f"HuggingFace agent initialized: {self.model_name}")
            return True
            
        except Exception as e:
            print(f"Error initializing HuggingFace agent: {e}")
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
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
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
        
        # Build final prompt - use Qwen's chat template if available, otherwise fallback
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            # Use proper chat template for Qwen
            messages = [
                {"role": "system", "content": f"{system}{history_text}{examples_text}"},
                {"role": "user", "content": user_input}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback format for models without chat templates
            prompt = f"<|system|>\n{system}{history_text}{examples_text}\n</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
        
        return prompt
    
    def generate_response(
        self,
        user_input: str,
        context: str = "casual",
        emotion: Optional[str] = None,
        include_examples: bool = True
    ) -> Dict:
        """Generate response using HuggingFace model"""
        
        if not self.model or not self.pipeline:
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
            
            # Generate response with parameters optimized for character consistency
            outputs = self.pipeline(
                prompt,
                max_new_tokens=80,  # Increased for better sentence completion
                do_sample=True,
                temperature=0.7,  # Lower temperature for more consistent character voice
                top_k=50,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Higher to avoid repetition
                no_repeat_ngram_size=3  # Prevent repeating 3-word phrases
            )
            
            # Extract response from generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the generated text
            if prompt in generated_text:
                nick_response = generated_text.split(prompt)[-1].strip()
            elif "<|assistant|>" in generated_text:
                nick_response = generated_text.split("<|assistant|>")[-1].strip()
            elif "<|im_start|>assistant" in generated_text:
                # Qwen format
                nick_response = generated_text.split("<|im_start|>assistant")[-1].strip()
                # Remove Qwen's end token if present
                nick_response = nick_response.split("<|im_end|>")[0].strip()
            else:
                nick_response = generated_text[len(prompt):].strip()
            
            # Remove special tokens
            nick_response = nick_response.split("</s>")[0].strip()
            nick_response = nick_response.split("<|im_end|>")[0].strip()  # Qwen end token
            nick_response = nick_response.split("<|")[0].strip()
            
            # Remove meta-text that indicates model confusion
            meta_patterns = [
                'Context:', 'Emotion:', 'Recent conversation:', 
                'Sure,', 'Certainly', "Here's", 'edited version', 
                'longer response', 'Example responses:'
            ]
            for pattern in meta_patterns:
                if pattern in nick_response:
                    nick_response = nick_response.split(pattern)[0].strip()
            
            # Stop at dialogue markers
            for marker in ['User:', 'user:', 'Nick:', 'nick:', f'{self.character_name}:' if self.character_name else 'Character:']:
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
                # Context-aware fallback
                fallbacks = {
                    'investigation': "Let me think about this.",
                    'combat': "Stay sharp.",
                    'emotional': "It's complicated.",
                    'greeting': "Hello there.",
                    'casual': "What do you need?"
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
            'device': self.device,
            'ready': self.model is not None,
            'type': 'huggingface'
        }
