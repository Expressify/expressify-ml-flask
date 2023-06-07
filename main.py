from flask import Flask, request, jsonify
from keras.models import load_model, model_from_json
import pickle
from keras.utils import pad_sequences
from PIL import Image
import numpy as np
from skimage import transform
from keras_preprocessing import image
import os
from keras.optimizers import Adam
from nltk.corpus import stopwords
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import regex as re
import cv2
import asyncio

nltk.download("stopwords")

# app setup
app = Flask(__name__)
app.config["LOCAL_FOLDER"] = os.getcwd() + "/upload/"
print(os.listdir(os.getcwd()))

# setup prediction
# indonesian slang word
slang_word = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "slang_word.txt"), sep="\t"
)
slang_dict = dict(slang_word.values)

# Stopword
stopword = stopwords.words("indonesian")
with open(os.path.join(os.path.dirname(__file__), "stopword_add_v2.txt")) as f:
    add_stopword = f.read().splitlines()
    f.close()
stopword.extend(add_stopword)

# Stemming
factory = StemmerFactory()

stemmer = factory.create_stemmer()


def text_correction(text):
    correct_word = []
    for word in text.split():
        # searching from slang_dict
        correct_word.append(slang_dict.get(word, word))
    correct_word = " ".join(correct_word)
    return correct_word


# preprocessing
def preprocessing(text):
    text = text.lower()  # case folding
    # cleaning
    text = re.sub(r"\d+", "", text)
    text = re.sub("((www\.[^\s]+)|(https?://[^\s]+)|\w+\.(com|id|org))", "", text)
    text = re.sub("(\n|\n\n|\xa0)", " ", text)
    text = re.sub("[•■®—«]", "", text)
    text = re.sub(r"(#|@)([^\s:]+)", "", text)  # remove tag account
    text = re.sub(
        "[^\w\s]", " ", text
    )  # remove punctuation, \w = alphanumeric, \s whitespace
    text = re.sub(" +", " ", text)
    text = re.sub("â€|ï", "", text)
    text = re.sub("\u200d", "", text)
    text = text_correction(text)
    text = stemmer.stem(text)
    text = " ".join(
        [word for word in text.split() if word not in (stopword)]
    )  # remove stopword

    text = text.strip()
    return text


# ml setup
try:
    # Load emotion detection model
    emotion_detection_model = load_model(
        os.path.join(os.path.dirname(__file__), "face_classifier_new.h5"), compile=False
    )

    # Load mental health prediction model
    json_file = open(os.path.join(os.path.dirname(__file__), "model_CNN.json"))
    loaded_model_json = json_file.read()
    json_file.close()
    mental_health_prediction_model = model_from_json(loaded_model_json)

    mental_health_prediction_model.load_weights(
        os.path.join(os.path.dirname(__file__), "model_CNN.h5")
    )

    with open(
        os.path.join(os.path.dirname(__file__), "tokenizer_CNN.pickle"), "rb"
    ) as handle:
        tokenizer_classification = pickle.load(handle)

except:
    print(os.listdir(os.getcwd()))

emotion_detection_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
pic_size = 224
label_dict = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@app.route("/emotion_prediction", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        print(image_file)
        image_path = os.path.join(app.config["LOCAL_FOLDER"], image_file.filename)
        image_file.save(image_path)
        test_image = load(image_path)
        print(test_image)
        prediction = test_image["prediction"]
        if prediction != "failed":
            return jsonify(prediction=prediction, status=True)
        else:
            return jsonify(status=False)
    if request.method == "GET":
        return "Server Up!"


@app.route("/jurnal_prediction", methods=["POST"])
def predict():
    if request.method == "POST":
        data = request.get_json()
        text = data["text"]
        text_cleaned = preprocessing(text)
        word_seq = tokenizer_classification.texts_to_sequences([text_cleaned])
        word_pad = pad_sequences(word_seq, maxlen=71)
        classes = ["Tidak Terindikasi Mental Illness", "Terindikasi Mental Illness"]
        predicted_class = mental_health_prediction_model.predict(word_pad).argmax(
            axis=1
        )
        return jsonify(prediction=classes[predicted_class[0]], status=True)


def load(filename):
    prediction = ""
    image = cv2.imread(filename)
    color = (0, 255, 0)
    pic_size = 224
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
    )
    for x, y, w, h in faces:
        # for each face on the image detected by OpenCV
        # get extended image of this face
        face_image = get_extended_image(image, x, y, w, h, 0.5)
        face_image = np.array(face_image).astype("float32") / 255
        face_image = transform.resize(face_image, (pic_size, pic_size, 3))
        face_image = np.expand_dims(face_image, axis=0)
        result = emotion_detection_model.predict(face_image)
        prediction = label_dict[np.array(result[0]).argmax(axis=0)]  # predicted class
        confidence = np.array(result[0]).max(axis=0)  # degree of confidence
    if prediction:
        return {"prediction": prediction, "confidence": confidence}
    else:
        return {"prediction": "failed"}


def get_extended_image(img, x, y, w, h, k=0.1):
    """
    Function, that return cropped image from 'img'
    If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
    If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
    After getting the desired image resize it to 250x250.
    And converts to tensor with shape (1, 250, 250, 3)

    Parameters:
        img (array-like, 2D): The original image
        x (int): x coordinate of the upper-left corner
        y (int): y coordinate of the upper-left corner
        w (int): Width of the desired image
        h (int): Height of the desired image
        k (float): The coefficient of expansion of the image

    Returns:
        image (tensor with shape (1, 224, 224, 3))
    """

    # The next code block checks that coordinates will be non-negative
    # (in case if desired image is located in top left corner)
    if x - k * w > 0:
        start_x = int(x - k * w)
    else:
        start_x = x
    if y - k * h > 0:
        start_y = int(y - k * h)
    else:
        start_y = y

    end_x = int(x + (1 + k) * w)
    end_y = int(y + (1 + k) * h)

    face_image = img[start_y:end_y, start_x:end_x]
    return face_image


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=(os.environ.get("PORT", 8080)))
