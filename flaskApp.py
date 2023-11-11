import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

pred_model = pickle.load(open("prediction_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])

def predict():
    data = request.get_json(force=True)
    input_txt = data["input"]

    separateInputByComma = input_txt.split(",") 
    df = np.asarray(separateInputByComma, dtype=float)
    prediction = pred_model.predict(df.reshape(1, -1))

    result = {
        "prediction": int(prediction[0])
    }

    return jsonify(result)


if __name__ == "__main__":
        app.run(port=5000)
