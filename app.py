from flask import Flask, render_template, request
from ChatApp import chatbot_response
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chat-app.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return render_template('chat-app.html', user_input=user_input, response=response)


if __name__ == '__main__':
    app.run(debug=True)
