"""
Model Comparison Tool
Compares the local instruct model with Gemini for character mimicry
"""
import os
import json
from typing import List, Dict
from agents.huggingface_agent import HuggingFaceAgent
from agents.gemini_agent import GeminiAgent


class ModelComparator:
    """Compare responses from different models"""
    
    def __init__(self):
        self.csv_path = os.path.join(os.path.dirname(__file__), 'data', 'nick_valentine_dialogue.csv')
        self.slm_agent = None
        self.llm_agent = None
        self.results = []
    
    def initialize_agents(self):
        """Initialize both agents"""
        print("Initializing agents for comparison...")
        
        # Initialize SLM (Qwen)
        print("\n1. Initializing local instruct model...")
        from model_defaults import DEFAULT_HF_MODEL

        self.slm_agent = HuggingFaceAgent(
            csv_path=self.csv_path,
            model_name=os.getenv("HF_MODEL", DEFAULT_HF_MODEL)
        )
        slm_ready = self.slm_agent.initialize()
        if not slm_ready:
            print("ERROR: Failed to initialize SLM agent")
            return False
        
        # Initialize LLM (Gemini)
        print("\n2. Initializing LLM (Gemini)...")
        self.llm_agent = GeminiAgent(
            csv_path=self.csv_path,
            model_name=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        )
        llm_ready = self.llm_agent.initialize()
        if not llm_ready:
            print("WARNING: Failed to initialize Gemini agent (check GEMINI_API_KEY)")
            print("Continuing with SLM-only comparison...")
        
        return True
    
    def run_comparison(
        self,
        prompts: List[Dict],
        reset_between: bool = True
    ) -> List[Dict]:
        """
        Run comparison tests
        
        Args:
            prompts: List of dicts with 'user_input', 'context', 'emotion', 'include_examples'
            reset_between: Whether to reset conversation between prompts
        
        Returns:
            List of comparison results
        """
        results = []
        
        for i, prompt_data in enumerate(prompts, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(prompts)}: {prompt_data['user_input']}")
            print(f"{'='*60}")
            
            if reset_between:
                self.slm_agent.reset_conversation()
                if self.llm_agent:
                    self.llm_agent.reset_conversation()
            
            # Generate SLM response
            print("\n[SLM] Generating response...")
            try:
                slm_result = self.slm_agent.generate_response(
                    user_input=prompt_data['user_input'],
                    context=prompt_data.get('context', 'casual'),
                    emotion=prompt_data.get('emotion'),
                    include_examples=prompt_data.get('include_examples', True)
                )
                slm_response = slm_result.get('response', 'ERROR')
                print(f"[SLM] Response: {slm_response}")
            except Exception as e:
                slm_response = f"ERROR: {str(e)}"
                print(f"[SLM] Error: {e}")
            
            # Generate LLM response (if available)
            llm_response = None
            if self.llm_agent:
                print("\n[LLM] Generating response...")
                try:
                    llm_result = self.llm_agent.generate_response(
                        user_input=prompt_data['user_input'],
                        context=prompt_data.get('context', 'casual'),
                        emotion=prompt_data.get('emotion'),
                        include_examples=prompt_data.get('include_examples', True)
                    )
                    llm_response = llm_result.get('response', 'ERROR')
                    print(f"[LLM] Response: {llm_response}")
                except Exception as e:
                    llm_response = f"ERROR: {str(e)}"
                    print(f"[LLM] Error: {e}")
            
            # Store result
            result = {
                'test_number': i,
                'prompt': prompt_data['user_input'],
                'context': prompt_data.get('context', 'casual'),
                'emotion': prompt_data.get('emotion'),
                'slm_response': slm_response,
                'llm_response': llm_response,
            }
            results.append(result)
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print comparison summary"""
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}\n")
        
        for result in self.results:
            print(f"Test {result['test_number']}: {result['prompt']}")
            print(f"  Context: {result['context']}, Emotion: {result.get('emotion', 'None')}")
            print(f"\n  SLM (Qwen 2.5 3B):")
            print(f"    {result['slm_response']}")
            if result['llm_response']:
                print(f"\n  LLM (Gemini):")
                print(f"    {result['llm_response']}")
            print("\n" + "-"*60 + "\n")
    
    def save_results(self, filename: str = "comparison_results.json"):
        """Save results to JSON file"""
        output_path = os.path.join(os.path.dirname(__file__), filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    def generate_markdown_report(self, filename: str = "COMPARISON_REPORT.md"):
        """Generate markdown comparison report"""
        output_path = os.path.join(os.path.dirname(__file__), filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Model Comparison: SLM vs LLM for Character Mimicry\n\n")
            f.write("## Overview\n\n")
            f.write("This report compares Small Language Model (SLM) performance against Large Language Model (LLM) ")
            f.write("for character dialogue generation, specifically for Nick Valentine from Fallout 4.\n\n")
            f.write("**Models Compared:**\n")
            f.write("- **SLM**: Qwen/Qwen2.5-3B-Instruct (3B parameters, local)\n")
            f.write("- **LLM**: Gemini 1.5 Flash (larger model, API-based)\n\n")
            f.write("**Key Hypothesis**: SLMs are better at staying in character because:\n")
            f.write("1. Less 'helpful assistant' behavior (big LLMs try to be helpful and break character)\n")
            f.write("2. More focused training (smaller models follow instructions more literally)\n")
            f.write("3. Less overthinking (big LLMs add unnecessary context or explanations)\n")
            f.write("4. Better at staying in character (less likely to explain things or break character)\n\n")
            f.write("---\n\n")
            
            for result in self.results:
                f.write(f"## Test {result['test_number']}: {result['prompt']}\n\n")
                f.write(f"**Context**: {result['context']}  ")
                f.write(f"**Emotion**: {result.get('emotion', 'None')}\n\n")
                f.write("### SLM Response (Qwen 2.5 3B)\n\n")
                f.write(f"> {result['slm_response']}\n\n")
                if result['llm_response']:
                    f.write("### LLM Response (Gemini)\n\n")
                    f.write(f"> {result['llm_response']}\n\n")
                f.write("---\n\n")
            
            f.write("## Analysis\n\n")
            f.write("### Why SLMs Excel at Character Mimicry\n\n")
            f.write("1. **Literal Instruction Following**: SLMs follow character instructions more literally without ")
            f.write("adding unnecessary context or explanations.\n\n")
            f.write("2. **Less Helpful Behavior**: Large LLMs are trained to be helpful assistants, which can ")
            f.write("cause them to break character by explaining things or being overly helpful.\n\n")
            f.write("3. **Focused Responses**: SLMs produce more focused, in-character responses without ")
            f.write("overthinking or adding meta-commentary.\n\n")
            f.write("4. **Consistency**: Smaller models are more consistent in following the same prompt structure ")
            f.write("without variation that breaks character.\n\n")
        
        print(f"\nMarkdown report saved to: {output_path}")


def main():
    """Run comparison tests"""
    
    # Test prompts for comparison
    test_prompts = [
        {
            'user_input': "Hello Nick",
            'context': 'greeting',
            'emotion': None,
            'include_examples': True
        },
        {
            'user_input': "I need help with a murder case",
            'context': 'investigation',
            'emotion': 'neutral',
            'include_examples': True
        },
        {
            'user_input': "What do you think of Piper?",
            'context': 'casual',
            'emotion': 'amused',
            'include_examples': True
        },
        {
            'user_input': "Do you ever feel alone?",
            'context': 'emotional',
            'emotion': 'somber',
            'include_examples': True
        },
        {
            'user_input': "The victim was found at the docks",
            'context': 'investigation',
            'emotion': 'neutral',
            'include_examples': True
        },
    ]
    
    comparator = ModelComparator()
    
    if not comparator.initialize_agents():
        print("Failed to initialize agents. Exiting.")
        return
    
    print("\n" + "="*60)
    print("Starting Comparison Tests")
    print("="*60)
    
    # Run comparisons
    results = comparator.run_comparison(test_prompts, reset_between=True)
    
    # Print summary
    comparator.print_summary()
    
    # Save results
    comparator.save_results()
    comparator.generate_markdown_report()
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

