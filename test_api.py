import sys
import os
import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"


def classify_image(image_path: str):
    base_name, image_extension = os.path.splitext(image_path)
    image_dir, image_name = os.path.split(image_path)

    if not os.path.isfile(image_path):
        raise FileExistsError(f"The '{image_name}' does not exists in the path.")
    
    if image_extension not in ['.jpg', '.png', '.jpeg']:
        raise ValueError("Expected only '.jpg' '.png', '.jpeg'")

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
