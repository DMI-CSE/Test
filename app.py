from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
import json

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' in request.files:
            uploaded_image = request.files['image']

            # Check if the file is empty

            # Read the image data from FileStorage and process it
            image_data = uploaded_image.read()
            image_path = os.path.join(
                r'./static', uploaded_image.filename)
            uploaded_image.save(image_path)
            image_stream = io.BytesIO(image_data)

            # Process the image
            model = InceptionV3(weights='imagenet')
            img = image.load_img(image_stream, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            # Make predictions
            predictions = model.predict(img)

            # Decode the predictions to human-readable labels
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            list_of_predictions = [label[1] for label in decoded_predictions]
            predicted_result = list_of_predictions[0]
            with open('./data.json', 'r') as data:
                d = json.load(data)
            if predicted_result not in d:
                predicted_result = 'not a dog'
            try:
                life_spwan = d[predicted_result]['lifespan']
                average_height = d[predicted_result]['average_height']
                average_weight = d[predicted_result]['average_weight']
                orgin = d[predicted_result]['origin']
            except:
                life_spwan = ''
                average_height = ""
                average_weight = ""
                orgin = ""
            return render_template("index.html", prediction=predicted_result, life=life_spwan, height=average_height, weight=average_weight, orgin=orgin, path=image_path)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
