import os
from urllib import response
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai 

app = Flask(__name__)

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

def get_completion(prompt):
      print("User prompt:", prompt)
      generation_config = {
        "temperature": 0.5,
        "max_output_tokens": 10000,
        "top_p": 0.95,
        "top_k": 64
    }
      response = model.generate_content(prompt, generation_config=generation_config)
      return response.text
  
  
@app.route("/", methods=['POST', 'GET'])
def index():
      if request.method == 'POST':
          print('Step 1')
          prompt = request.form['prompt']
          response = get_completion(prompt)
          print('Response : ', response)
          return jsonify({'response': response})
      
      return render_template("index.html")
  
  

if __name__ == "__main__":
    app.run(debug=True)  




