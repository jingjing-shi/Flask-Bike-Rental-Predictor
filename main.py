
import numpy as np
from flask import Flask, request, jsonify, render_template
import xgboost

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    model = xgboost.XGBRegressor()
    model.load_model('model.json')
    float_features = [np.float(x) for x in request.form.values()]
    final_features = np.array(float_features).reshape(1,13)
    prediction = model.predict(final_features)

    output = int(prediction[0])

    return render_template('index.html', prediction_text='Bike Sharing Demand is {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
