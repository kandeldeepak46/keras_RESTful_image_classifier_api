# keras_REST_image_classifier
sample api for image classifier using keras and Flask

# Usage
```bash
$ git clone git@github.com:deepak-kandel/keras_REST_image_classifier.git
$ cd keras_REST_image_classifier
$ virtualenv --no-site-packages venv
$ source venv/Scripts/activate
$ pip install -r requirements.txt
```
## Run the local server
```bash
$ python app.py
```
### Test the image  via command line arguments
```bash
$ python test_api.py images/cat.jpg
```
### Test the image via notebook
```bash
$ from test_api import classify_image
$ classify_image('images/cat.jpg')
