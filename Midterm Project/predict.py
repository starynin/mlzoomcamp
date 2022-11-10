# Load the model
import pickle
#from platform import machine

from flask import Flask
from flask import request
from flask import jsonify

input_file = 'project1_model_FR.bin'

with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

app = Flask('PredictiveMaintenance')

@app.route('/predict', methods=['POST'])
def predict():
    tool = request.get_json()

    X = dv.transform([tool])
    y_pred = model.predict_proba(X)[0, 1]
    fail = y_pred >= 0.5

    result = {
        'failure_probability': float(y_pred),
        'failure': bool(fail)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
