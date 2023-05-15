from flask import Flask, request
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform
from keras_preprocessing import image
import os

# app setup
app = Flask(__name__)
app.config['LOCAL_FOLDER'] = os.getcwd() + '/upload/'

# ml setup
emotion_detection_model = load_model('face_classifier_new.h5')
pic_size = 224
label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

print(emotion_detection_model)

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = os.path.join(app.config['LOCAL_FOLDER'], image_file.filename)
        image_file.save(image_path)
        test_image = load(image_path)
        result = emotion_detection_model.predict(test_image)
        result = list(result[0])
        img_index = result.index(max(result))
        return label_dict[img_index]


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (pic_size, pic_size, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
if __name__ == "__main__":
    app.run(debug=True)