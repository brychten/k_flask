import datetime
import uuid

from flask import request, Flask
import pandas as pd 
from ms.functions import get_model_response

# Initialize App
app = Flask(__name__)

# Load models
model_name = "Breast Cancer Wisconsin (Diagnostic)"
model_file = "model_binary.dat.gz"
#
version = "v1.0.0"
instance_id = uuid.uuid4().hex


@app.route("/")
def get_instance_id():
    return f"Instance ID: {instance_id}"

@app.route("/info", methods=["GET"])
def info():
    "Return model information"
    result = {}
    result["name"] = model_name
    result["version"] = version

    return result

@app.route('/health', methods=['GET'])
def health():
    """Return service health"""
    return 'ok'

@app.route('/predict', methods=['POST'])
def predict():
    print(request.get_json())
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
        }, 500

    try:
        response = get_model_response(feature_dict)
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500

    return response, 200




if __name__ == "__main__":
    app.run(host="0.0.0.0")