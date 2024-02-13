from flask import Flask, render_template, request
import numpy as np
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
app = Flask(__name__)

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

model = ResNet50(weights='imagenet')

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/upload',methods=['POST'])
def upload():
    filename = photos.save(request.files['image'])
    img_path = './static/img/' + filename

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) 

    preds = model.predict(x)
    pred_class = decode_predictions(preds, top=1)[0][0]
    class_name = pred_class[1]
    class_probability = pred_class[2] * 100
    return render_template('input.html',resultat=f"{class_name} sur à {class_probability:.2f}%")


if __name__ == '__main__':
    app.run()