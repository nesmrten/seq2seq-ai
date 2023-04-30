from flask import Flask, render_template, request, jsonify
from models.chatbot import Chatbot
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

chatbot = Chatbot(model_file=app.config['MODEL_FILE'], tokenizer_file=app.config['TOKENIZER_FILE'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def chatbot_response():
    msg = request.args.get('msg')
    response = chatbot.reply(msg)
    return jsonify({'msg': response})

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])
