// Nick Valentine Dialogue Generator - Frontend

const elements = {
  conversationDisplay: document.getElementById('conversationDisplay'),
  userInput: document.getElementById('userInput'),
  generateBtn: document.getElementById('generateBtn'),
  resetBtn: document.getElementById('resetBtn'),
  exportBtn: document.getElementById('exportBtn'),
  agentSelect: document.getElementById('agentSelect'),
  agentInfo: document.getElementById('agentInfo'),
  contextSelect: document.getElementById('contextSelect'),
  emotionSelect: document.getElementById('emotionSelect'),
  examplesCheck: document.getElementById('examplesCheck'),
  messageCount: document.getElementById('messageCount'),
  costEstimate: document.getElementById('costEstimate'),
  status: document.getElementById('status'),
  btnText: document.getElementById('btnText'),
  btnLoader: document.getElementById('btnLoader'),
};

let messageCounter = 0;
let totalCost = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  elements.generateBtn.addEventListener('click', generateDialogue);
  elements.resetBtn.addEventListener('click', resetConversation);
  elements.exportBtn.addEventListener('click', exportConversation);
  elements.agentSelect.addEventListener('change', switchAgent);
  
  // Allow Enter to submit (with Shift+Enter for new line)
  elements.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      generateDialogue();
    }
  });
  
  // Load available agents
  loadAgents();
  
  // Focus on input
  elements.userInput.focus();
});

// Generate dialogue
async function generateDialogue() {
  const userInput = elements.userInput.value.trim();
  
  if (!userInput) {
    showNotification('Please enter something to say', 'error');
    return;
  }
  
  // Disable input while generating
  setGenerating(true);
  
  // Add player message to display
  addMessage(userInput, 'player');
  
  // Clear input
  elements.userInput.value = '';
  
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userInput,
        context: elements.contextSelect.value,
        emotion: elements.emotionSelect.value || null,
        includeExamples: elements.examplesCheck.checked,
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to generate response');
    }
    
    const data = await response.json();
    
    // Add Nick's response
    addMessage(data.response, 'nick', {
      context: data.context,
      emotion: data.emotion,
      fallback: data.fallback,
    });
    
    // Update stats
    messageCounter++;
    updateStats();
    
    if (data.error) {
      showNotification(`API Error: ${data.error}`, 'warning');
    }
    
  } catch (error) {
    console.error('Error:', error);
    addMessage("Something's not right here... [Error connecting to server]", 'nick', { error: true });
    showNotification('Failed to generate response', 'error');
  } finally {
    setGenerating(false);
    elements.userInput.focus();
  }
}

// Add message to conversation display
function addMessage(text, type, meta = {}) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}-message`;
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  
  const textP = document.createElement('p');
  
  if (type === 'player') {
    textP.innerHTML = `<strong>You:</strong> ${escapeHtml(text)}`;
  } else if (type === 'nick') {
    textP.innerHTML = `<strong>Nick:</strong> ${escapeHtml(text)}`;
  } else {
    textP.innerHTML = `<strong>System:</strong> ${escapeHtml(text)}`;
  }
  
  contentDiv.appendChild(textP);
  
  // Add metadata if present
  if (Object.keys(meta).length > 0 && type === 'nick') {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    const metaParts = [];
    
    if (meta.context) metaParts.push(`Context: ${meta.context}`);
    if (meta.emotion) metaParts.push(`Emotion: ${meta.emotion}`);
    if (meta.fallback) metaParts.push('⚠️ Fallback response');
    if (meta.error) metaParts.push('❌ Error');
    
    metaDiv.textContent = metaParts.join(' • ');
    contentDiv.appendChild(metaDiv);
  }
  
  messageDiv.appendChild(contentDiv);
  elements.conversationDisplay.appendChild(messageDiv);
  
  // Scroll to bottom
  elements.conversationDisplay.scrollTop = elements.conversationDisplay.scrollHeight;
}

// Reset conversation
async function resetConversation() {
  if (!confirm('Are you sure you want to reset the conversation?')) {
    return;
  }
  
  try {
    const response = await fetch('/api/reset', {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error('Failed to reset conversation');
    }
    
    // Clear display
    elements.conversationDisplay.innerHTML = '';
    addMessage('Nick Valentine is ready to talk. What do you want to say?', 'system');
    
    // Reset stats
    messageCounter = 0;
    totalCost = 0;
    updateStats();
    
    showNotification('Conversation reset', 'success');
    
  } catch (error) {
    console.error('Error:', error);
    showNotification('Failed to reset conversation', 'error');
  }
}

// Export conversation
async function exportConversation() {
  try {
    const response = await fetch('/api/export');
    
    if (!response.ok) {
      throw new Error('Failed to export conversation');
    }
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `nick-conversation-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showNotification('Conversation exported', 'success');
    
  } catch (error) {
    console.error('Error:', error);
    showNotification('Failed to export conversation', 'error');
  }
}

// Update stats display
async function updateStats() {
  elements.messageCount.textContent = messageCounter;
  
  // Fetch cost estimate
  try {
    const response = await fetch(`/api/cost-estimate?exchanges=${messageCounter}`);
    if (response.ok) {
      const data = await response.json();
      elements.costEstimate.textContent = data.estimatedCost;
    }
  } catch (error) {
    console.error('Error fetching cost estimate:', error);
  }
}

// Set generating state
function setGenerating(isGenerating) {
  elements.generateBtn.disabled = isGenerating;
  elements.userInput.disabled = isGenerating;
  
  if (isGenerating) {
    elements.btnText.textContent = 'Generating...';
    elements.btnLoader.style.display = 'inline-block';
    elements.status.textContent = 'Generating';
    elements.status.className = 'stat-value status-generating';
  } else {
    elements.btnText.textContent = 'Generate Response';
    elements.btnLoader.style.display = 'none';
    elements.status.textContent = 'Ready';
    elements.status.className = 'stat-value status-ready';
  }
}

// Show notification
function showNotification(message, type = 'info') {
  // Simple console notification for now
  // You can enhance this with a proper toast notification system
  console.log(`[${type.toUpperCase()}] ${message}`);
  
  // Optionally add a temporary message to the conversation
  if (type === 'error') {
    const tempMsg = document.createElement('div');
    tempMsg.className = 'message system-message';
    tempMsg.innerHTML = `<div class="message-content"><p><strong>System:</strong> ${escapeHtml(message)}</p></div>`;
    tempMsg.style.opacity = '0.6';
    elements.conversationDisplay.appendChild(tempMsg);
    
    setTimeout(() => {
      tempMsg.remove();
    }, 3000);
  }
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Load available agents
async function loadAgents() {
  try {
    const response = await fetch('/api/agents');
    if (!response.ok) throw new Error('Failed to load agents');
    
    const data = await response.json();
    const agents = data.agents;
    
    // Populate agent dropdown
    elements.agentSelect.innerHTML = '';
    agents.forEach(agent => {
      const option = document.createElement('option');
      option.value = agent.id;
      option.textContent = `${agent.name} (${agent.model})`;
      if (agent.active) {
        option.selected = true;
      }
      elements.agentSelect.appendChild(option);
    });
    
    // Update agent info
    updateAgentInfo();
    
  } catch (error) {
    console.error('Error loading agents:', error);
    elements.agentInfo.textContent = 'Agent: Error loading';
  }
}

// Switch agent
async function switchAgent() {
  const agentId = elements.agentSelect.value;
  
  if (!agentId) return;
  
  try {
    const response = await fetch('/api/agents/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentId }),
    });
    
    if (!response.ok) throw new Error('Failed to switch agent');
    
    const data = await response.json();
    
    // Clear conversation display
    elements.conversationDisplay.innerHTML = '';
    addMessage(`Switched to ${agentId}. Starting fresh conversation.`, 'system');
    
    // Reset stats
    messageCounter = 0;
    updateStats();
    updateAgentInfo();
    
    showNotification(data.message, 'success');
    
  } catch (error) {
    console.error('Error switching agent:', error);
    showNotification('Failed to switch agent', 'error');
  }
}

// Update agent info display
async function updateAgentInfo() {
  try {
    const response = await fetch('/api/agents/active');
    if (!response.ok) return;
    
    const data = await response.json();
    elements.agentInfo.textContent = `Agent: ${data.name || 'Unknown'} | Model: ${data.model || 'Unknown'}`;
    
  } catch (error) {
    console.error('Error updating agent info:', error);
  }
}

// Initial stats update
updateStats();
