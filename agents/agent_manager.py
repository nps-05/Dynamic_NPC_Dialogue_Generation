"""
Agent Manager
Manages multiple dialogue generation agents and handles switching between them
"""

from typing import Dict, Optional, List
from agents.base_agent import BaseDialogueAgent


class AgentManager:
    """Manages multiple dialogue agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseDialogueAgent] = {}
        self.active_agent: Optional[str] = None
    
    def register_agent(self, agent_id: str, agent: BaseDialogueAgent):
        """Register a new agent"""
        self.agents[agent_id] = agent
        print(f"✓ Registered agent: {agent_id} ({agent.name})")
        
        # Set as active if it's the first agent
        if not self.active_agent:
            self.active_agent = agent_id
            print(f"✓ Set {agent_id} as active agent")
    
    def switch_agent(self, agent_id: str) -> Dict:
        """Switch to a different agent (resets conversation)"""
        if agent_id not in self.agents:
            return {
                'success': False,
                'error': f"Agent '{agent_id}' not found"
            }
        
        old_agent = self.active_agent
        self.active_agent = agent_id
        
        # Reset the new agent's conversation
        self.agents[agent_id].reset_conversation()
        
        return {
            'success': True,
            'old_agent': old_agent,
            'new_agent': agent_id,
            'message': f"Switched from {old_agent} to {agent_id}. Conversation reset."
        }
    
    def get_active_agent(self) -> Optional[BaseDialogueAgent]:
        """Get the currently active agent"""
        if self.active_agent and self.active_agent in self.agents:
            return self.agents[self.active_agent]
        return None
    
    def generate_response(
        self, 
        user_input: str, 
        context: str = "casual",
        emotion: Optional[str] = None,
        include_examples: bool = True
    ) -> Dict:
        """Generate response using the active agent"""
        agent = self.get_active_agent()
        
        if not agent:
            return {
                'response': "No active agent available.",
                'error': "No agent is currently active",
            }
        
        return agent.generate_response(user_input, context, emotion, include_examples)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history from active agent"""
        agent = self.get_active_agent()
        if agent:
            return agent.get_history()
        return []
    
    def reset_conversation(self) -> Dict:
        """Reset conversation for active agent"""
        agent = self.get_active_agent()
        if agent:
            agent.reset_conversation()
            return {
                'success': True,
                'message': f"Conversation reset for {agent.name}"
            }
        return {
            'success': False,
            'error': "No active agent"
        }
    
    def list_agents(self) -> List[Dict]:
        """List all registered agents"""
        return [
            {
                'id': agent_id,
                'name': agent.name,
                'model': agent.model_name,
                'active': agent_id == self.active_agent,
                'history_length': len(agent.conversation_history),
            }
            for agent_id, agent in self.agents.items()
        ]
    
    def get_agent_info(self, agent_id: Optional[str] = None) -> Dict:
        """Get information about a specific agent or the active one"""
        if agent_id:
            agent = self.agents.get(agent_id)
        else:
            agent = self.get_active_agent()
        
        if agent:
            return agent.get_info()
        return {}
