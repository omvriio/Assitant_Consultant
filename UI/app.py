from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_post():
    data = request.json
    user_message = data.get('message')
    # Here you can integrate your chatbot logic
    bot_response = f"Echo: {user_message}"
    return jsonify({'response': bot_response})

@app.route('/chat', methods=['GET'])
def chat_get():
    color_scheme = request.args.get('darkschemeovr') or request.args.get('lightschemeovr')
    set_lang = request.args.get('setlang')
    extension_id = request.args.get('extension')
    # Render an HTML template for the chat interface
    return render_template('index.html', color_scheme=color_scheme, set_lang=set_lang, extension_id=extension_id)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)