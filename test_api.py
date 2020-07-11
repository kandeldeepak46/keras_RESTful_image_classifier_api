import sys
import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"


def classify_image(image_path: str):
    try:
        with open(image_path, "rb") as i:
            image = i.read()
    except FileExistsError as e:
        raise FileExistsError("Image file not found.please check input")

    payload = {"image": image}
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    if r["success"]:
        for (i, result) in enumerate(r["predictions"]):
            print(f"{i + 1},{result['label']},{result['probability']}")
    else:
        print("Request failed")


def main():
    if len(sys.argv) == 1:
        raise SystemExit("Expected the image file. Exiting the process")
    if len(sys.argv) > 2 :
        raise SystemExit("Expected only one image. Exiting the process")
    return classify_image(sys.argv[1])


if __name__ == "__main__":
    main()
