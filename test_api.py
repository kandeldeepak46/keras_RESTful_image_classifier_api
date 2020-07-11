import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "images/dog.jpg"

image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

r = requests.post(KERAS_REST_API_URL, files=payload).json()


if r["success"]:
    for (i, result) in enumerate(r["predictions"]):
        print(f"{i + 1},{result['label']},{result['probability']}")

else:
    print("Request failed")
