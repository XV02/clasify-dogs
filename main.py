import pickle
from PIL import Image
import requests
import numpy as np
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import tensorflow_hub as hub
import tensorflow as tf
import Orange
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
from fastapi import FastAPI, File, UploadFile
import csv
import os

with open('clases.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

lr = pickle.load(open('Proyect-Model.pkcls', 'rb'))

app = FastAPI()

@app.post("/clasify")
def clasify_image(file: UploadFile):
    # Check if the file exists
    if not file:
        return {"Error": "No file found"}
    # Read the image via file.stream
    image = Image.open(file.file)
    # Save the image to ./uploads
    image.save(f"./uploads/{file.filename}")

    image_file_paths = [f"./uploads/{file.filename}"]

    print('Image uploaded')

    with ImageEmbedder(model='inception-v3') as emb:
        embeddings = emb(image_file_paths)
    
    print('Embeddings done')

    prediction = lr.predict(embeddings)

    prediction_array = np.array(prediction[1])

    print(data[np.argmax(prediction_array)][0])  

    # Delete the image from the server
    os.remove(f"./uploads/{file.filename}")

    return {"prediction": data[np.argmax(prediction_array)][0]} 
