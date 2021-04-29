
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])

    return render_template('index.html', prediction_text='Bike Sharing Demand is {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
