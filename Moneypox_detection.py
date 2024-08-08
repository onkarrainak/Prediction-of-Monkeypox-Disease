import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py

st.header("MlP_Predictive_model")
def main():
    file_uploaded = st.file_uploader("choose the file", type = ["jpg", "png", "jpeg"])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)
        
        
def predict_class(test_image):
    classifier_model = tf.keras.models.load_model("model1streamlit run project_1.py.h5")
    shape = (300, 300, 3)
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
    test_image = test_image.resize((300, 300))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image.astype('float32')/255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ["Chickenpox","Measles", "Monkeypox","Normal" ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result
    
if __name__ == "__main__":
    main()

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
