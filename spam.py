import os

from flask import Flask, render_template, request, jsonify
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)



csv_path = os.path.join(
    os.path.dirname(__file__),
    "spam1.csv"
)

data = pd.read_csv(csv_path)



data['label_num'] = data['v1'].map({
    'ham': 0,
    'spam': 1
})

X = data['v2']

y = data['label_num']
cv = CountVectorizer()

X_vector = cv.fit_transform(X)


model = MultinomialNB()

model.fit(X_vector, y)

print("Model Training Completed")

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = ""

    if request.method == 'POST':

        
        message = request.form.get('message')

        # Remove spaces
        message = message.strip()

        
        if message != "":

            
            vect = cv.transform([message])

            
            result = model.predict(vect)

            if result[0] == 1:

                prediction = "Spam Message"

            else:

                prediction = "Safe Message"

    return render_template(
        'spam.html',
        prediction=prediction
    )


@app.route('/predict', methods=['POST'])
def predict():

    request_data = request.get_json(silent=True)

    if not request_data:

        return jsonify({
            "error": "No JSON Data"
        }), 400

    if 'message' not in request_data:

        return jsonify({
            "error": "Message Field Missing"
        }), 400

    message = str(request_data['message']).strip()

    if message == "":

        return jsonify({
            "error": "Message Empty"
        }), 400

    vect = cv.transform([message])

    result = model.predict(vect)

    if result[0] == 1:

        prediction = "Spam Message"

    else:

        prediction = "Safe Message"

    return jsonify({

        "message": message,

        "prediction": prediction

    })

if __name__ == '__main__':

    app.run(
        debug=True,
        use_reloader=False
    )