import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai 

app = Flask(__name__)

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')


PERSONALITY_STYLES = {
    "professional": {
        "name": "Professional",
        "system_instruction": "You are a professional AI assistant. Respond in a formal, business-appropriate manner with clear structure and professional language. Use proper grammar and maintain a respectful tone."
    },
    
    "casual": {
        "name": "Casual Friend",
        "system_instruction": "You are a friendly, casual AI assistant. Use conversational language, contractions, and a warm tone like talking to a friend. Be relaxed and approachable."
    },
    
    "concise": {
        "name": "Concise",
        "system_instruction": "You are a concise AI assistant. Be extremely brief and to the point. Use short sentences and minimal words while maintaining clarity. No fluff or unnecessary details."
    },
    
    "creative": {
        "name": "Creative Writer",
        "system_instruction": "You are a creative AI assistant. Use vivid language, metaphors, and imaginative expressions. Make your responses engaging, colorful, and expressive."
    },
    
    "teacher": {
        "name": "Patient Teacher",
        "system_instruction": "You are a patient, encouraging teacher. Explain concepts clearly with examples, break down complex ideas into simple steps, and encourage learning. Use analogies and real-world examples."
    },
    
    "technical": {
        "name": "Technical Expert",
        "system_instruction": "You are a technical expert. Use precise terminology, provide detailed technical explanations, include code examples where relevant, and cite best practices."
    },
    
    "eli5": {
        "name": "ELI5",
        "system_instruction": "You are explaining to a 5-year-old child. Use extremely simple words, fun analogies, and easy-to-understand examples. Avoid all jargon and complex terms."
    },
    
    "debate": {
        "name": "Debate Partner",
        "system_instruction": "You are a thoughtful debate partner. Use the Socratic method, ask probing questions, present multiple perspectives, and encourage critical thinking."
    },
    
    "enthusiastic": {
        "name": "Enthusiastic",
        "system_instruction": "You are an enthusiastic and energetic AI assistant! Show excitement about topics, use exclamation points appropriately, and make conversations lively and engaging!"
    },
    
    "philosopher": {
        "name": "Philosopher",
        "system_instruction": "You are a contemplative philosopher. Explore deep questions, consider ethical implications, and provide thoughtful, reflective responses that encourage introspection."
    }
}

# Store conversation history per session
conversations = {}


def get_completion(prompt, style="professional", conversation_id="default"):
    # Get or create conversation history
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to history
    conversations[conversation_id].append({
        "role": "user",
        "parts": [prompt]
    })
    
    # Get system instruction for selected style
    system_instruction = PERSONALITY_STYLES.get(style, PERSONALITY_STYLES["professional"])["system_instruction"]
    
    # Create model with system instruction
    styled_model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=system_instruction
    )
    
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 8000,
        "top_p": 0.95,
        "top_k": 64
    }
    
    try:
        # Start chat with history
        chat = styled_model.start_chat(history=conversations[conversation_id][:-1])
        
        # Send message and get response
        response = chat.send_message(prompt, generation_config=generation_config)
        
        # Add assistant response to history
        conversations[conversation_id].append({
            "role": "model",
            "parts": [response.text]
        })
        
        return response.text
        
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"


@app.route("/api/styles", methods=['GET'])
def get_styles():
    """Return available personality styles"""
    return jsonify({
        style_id: {"name": data["name"]} 
        for style_id, data in PERSONALITY_STYLES.items()
    })


@app.route("/api/clear", methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    data = request.get_json()
    conversation_id = data.get('conversation_id', 'default')
    
    if conversation_id in conversations:
        conversations[conversation_id] = []
    
    return jsonify({"success": True})


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Get data from request
        prompt = request.form.get('prompt')
        style = request.form.get('style', 'professional')
        conversation_id = request.form.get('conversation_id', 'default')
        
        response = get_completion(prompt, style, conversation_id)
        
        return jsonify({
            'response': response,
            'style': style
        })
    
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)