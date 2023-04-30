from flask import Flask, request, render_template
from models.chatbot import Chatbot

app = Flask(__name__)

chatbot = Chatbot(model_file='models/model.pth', tokenizer_file='models/tokenizer.json')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response")
def get_bot_response():
    user_query = request.args.get('msg')
    return chatbot.generate_response(user_query)

if __name__ == "__main__":
    app.run(debug=True)
