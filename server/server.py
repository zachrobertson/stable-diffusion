import torch

from model import txt2img
from flask import Flask, request

app = Flask(__name__)

@app.route("/healthCheck", methods=["GET"])
def healthcheck():
    gpu = False
    if torch.cuda.is_available():
        gpu = True

    return {"state": "healthy", "gpu": gpu}


@app.route("/txt2img", methods=["POST"])
def inference():
    body = request.get_json()
    try:
        assert isinstance(body["inputs"], list) and all([isinstance(i, str) for i in body["inputs"]])
    except Exception as e:
        print("Inputs are not the correct type, must be List[str]")
        raise e

    response = txt2img(body["inputs"])
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)