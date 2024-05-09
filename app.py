import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    specs = [float(x) for x in request.form.values()]
    # Convert the data to a numpy array
    specs_array = np.array(specs).reshape(1, -1)
    # Make prediction using the loaded model
    predicted_price = model.predict(specs_array)
    return render_template('index.html', prediction_text='Predicted Price Catergory: {}'.format(predicted_price[0]))

if __name__ == '__main__':
    app.run(debug=True)
