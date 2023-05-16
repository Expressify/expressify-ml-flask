from flask import Flask, request
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform
from keras_preprocessing import image
import os
from keras.optimizers import Adam

# app setup
app = Flask(__name__)
app.config["LOCAL_FOLDER"] = os.getcwd() + "/upload/"
print(os.listdir(os.getcwd()))

# ml setup
try:
    emotion_detection_model = load_model(
        os.path.join(os.path.dirname(__file__), "face_classifier_new.h5"), compile=False
    )
except:
    print(os.listdir(os.getcwd()))

emotion_detection_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
pic_size = 224
label_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        image_path = os.path.join(app.config["LOCAL_FOLDER"], image_file.filename)
        image_file.save(image_path)
        test_image = load(image_path)
        result = emotion_detection_model.predict(test_image)
        result = list(result[0])
        img_index = result.index(max(result))
        return label_dict[img_index]
    if request.method == "GET":
        return "Server Up!"


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (pic_size, pic_size, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=(os.environ.get("PORT", 8080)))
