from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def predict_grade(marks):
    if marks >= 90:
        return "A"
    elif marks <90 and marks>=70:
        return "B"
    elif marks <70 and marks >= 50:
        return "C"
    elif marks <50 and marks >= 30:
        return "D"
    elif marks <30 and marks >= 20:
        return "E"
    else:
        return "F"

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg

    # Check if the input contains marks or relevant keywords
    if any(word in input_text.lower() for word in ["marks", "score"]) or any(char.isdigit() for char in input_text):
        # Extract marks from the user input
        marks = int(''.join(filter(str.isdigit, input_text)))
        # Predict grade
        predicted_grade = predict_grade(marks)
        response = f"Your grade is {predicted_grade}"
    else:
        response = get_chat_response(input_text)
    
    return response

def get_chat_response(text):
    # Let's chat for 5 lines
    for step in range(5):
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run()
