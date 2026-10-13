"""
Flask server for dynamic NPC dialogue generation
One local model, characters discovered from CSV and ChromaDB pairs
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os

from agents.agent_manager import AgentManager
from characters import discover_characters, public_character
from model_defaults import DEFAULT_HF_MODEL

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='public', static_url_path='')
CORS(app)

# Initialize Agent Manager
agent_manager = AgentManager()
_agents_initialized = False

# Initialize agents
def initialize_agents():
    """Initialize all available agents"""
    global _agents_initialized
    
    # Prevent re-initialization during Flask reloader
    if _agents_initialized:
        print("Agents already initialized, skipping...")
        return
    
    # The debug reloader imports this file twice. Only the child should load the model.
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    if debug and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print("Waiting for Flask reloader...")
        return
    
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'nick_valentine_dialogue.csv')
        
        # HuggingFace agent. Model id comes from HF_MODEL or model_defaults.DEFAULT_HF_MODEL.
        # HF_TOKEN is optional - Qwen models don't require authentication
        hf_token = os.getenv('HF_TOKEN')  # Optional
        
        try:
            from agents.huggingface_agent import HuggingFaceAgent

            model_name = os.getenv('HF_MODEL', DEFAULT_HF_MODEL)
            
            print(f"Initializing HuggingFace agent ({model_name})...")
            if not hf_token:
                print("   (Running without HF_TOKEN - Qwen models don't require authentication)")
            
            hf_agent = HuggingFaceAgent(
                csv_path=csv_path,
                model_name=model_name
            )
            init_result = hf_agent.initialize()
            if init_result:
                agent_manager.register_agent('huggingface', hf_agent)
                characters = discover_characters()
                if characters:
                    hf_agent.use_character(characters[0])
                    print(f"Active character: {characters[0]['name']}")
                else:
                    print("No character with both a CSV and a vector database. Build one to enable chat.")
                print(f"HuggingFace agent ({model_name}) initialized successfully!")
                _agents_initialized = True
            else:
                print("Error: HuggingFace agent initialization returned False - check errors above")
        except Exception as e:
            print(f"Error: Failed to initialize HuggingFace agent: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Critical error during initialization: {e}")
        import traceback
        traceback.print_exc()


# Routes

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('public', 'index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate dialogue response using active agent"""
    data = request.json
    
    user_input = data.get('userInput', '').strip()
    if not user_input:
        return jsonify({'error': 'userInput is required'}), 400
    
    context = data.get('context', 'casual')
    emotion = data.get('emotion')
    include_examples = data.get('includeExamples', True)
    
    try:
        response = agent_manager.generate_response(
            user_input=user_input,
            context=context,
            emotion=emotion,
            include_examples=include_examples
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agents', methods=['GET'])
def list_agents():
    """List all available agents"""
    agents = agent_manager.list_agents()
    return jsonify({'agents': agents})


@app.route('/api/agents/switch', methods=['POST'])
def switch_agent():
    """Switch to a different agent"""
    data = request.json
    agent_id = data.get('agentId')
    
    if not agent_id:
        return jsonify({'error': 'agentId is required'}), 400
    
    result = agent_manager.switch_agent(agent_id)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 404


@app.route('/api/agents/active', methods=['GET'])
def get_active_agent():
    """Get information about the active agent"""
    info = agent_manager.get_agent_info()
    return jsonify(info)


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history from active agent"""
    history = agent_manager.get_conversation_history()
    return jsonify({'history': history})


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset conversation for active agent"""
    result = agent_manager.reset_conversation()
    return jsonify(result)


@app.route('/api/export', methods=['GET'])
def export():
    """Export conversation history"""
    history = agent_manager.get_conversation_history()
    agent = agent_manager.get_active_agent()
    
    export_data = {
        'agent': agent.name if agent else 'Unknown',
        'model': agent.model_name if agent else 'Unknown',
        'conversation': history
    }
    
    return jsonify(export_data)


@app.route('/api/cost-estimate', methods=['GET'])
def cost_estimate():
    """Get cost estimate for number of exchanges"""
    exchanges = int(request.args.get('exchanges', 100))
    
    # Qwen model runs locally and is completely FREE
    return jsonify({
        'exchanges': exchanges,
        'estimatedCost': '$0.00',
        'note': 'The local model has no API cost.',
        'model': os.getenv('HF_MODEL', DEFAULT_HF_MODEL)
    })


@app.route('/api/characters', methods=['GET'])
def list_characters():
    """Characters that currently have both a CSV and a Chroma database."""
    characters = discover_characters()
    agent = agent_manager.get_active_agent()
    active = getattr(agent, 'character_id', None) if agent else None
    return jsonify({
        'characters': [public_character(character) for character in characters],
        'active': active,
    })


@app.route('/api/characters/switch', methods=['POST'])
def switch_character():
    """Switch the active agent to another discovered character."""
    data = request.json or {}
    character_id = data.get('characterId')
    if not character_id:
        return jsonify({'error': 'characterId is required'}), 400

    match = next((item for item in discover_characters() if item['id'] == character_id), None)
    if not match:
        return jsonify({
            'error': 'Character not found. The CSV and vector database must both exist.',
        }), 404

    agent = agent_manager.get_active_agent()
    if not agent or not hasattr(agent, 'use_character'):
        return jsonify({'error': 'No dialogue agent is ready'}), 503

    agent.use_character(match)
    return jsonify({
        'success': True,
        'character': public_character(match),
        'message': f"Switched to {match['name']}. Conversation reset.",
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'agents_count': len(agent_manager.agents),
        'active_agent': agent_manager.active_agent,
    })


if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 3000))
    # Only the reloader child should load the model. A normal run has no child.
    if not debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("\nDynamic NPC Dialogue (Python/Flask)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        initialize_agents()
        if not agent_manager.agents:
            print("Error: No agents initialized. The server will start, but generation will fail.")
        else:
            print(f"Active agent: {agent_manager.active_agent}")
        print(f"Server running at http://{host}:{port}")
        print("   GET    /api/characters")
        print("   POST   /api/characters/switch")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    app.run(host=host, port=port, debug=debug, use_reloader=debug)
