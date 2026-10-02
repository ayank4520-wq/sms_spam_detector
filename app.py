from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the model ONCE when the server starts — not on every request.
# This is the key difference between "training" and "serving."
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Please enter a message'}), 400

    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]

    # predict_proba gives confidence, not just a yes/no label
    probabilities = model.predict_proba(vec)[0]
    spam_index = list(model.classes_).index('spam')
    spam_probability = round(float(probabilities[spam_index]) * 100, 1)

    return jsonify({
        'prediction': prediction,
        'spam_probability': spam_probability
    })


if __name__ == '__main__':
    app.run(debug=True)
