import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import tensorflow


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@st.cache_resource
def load_features():
    return np.array(pickle.load(open('list.pkl', 'rb'))), pickle.load(open('files.pkl', 'rb'))

feature_list, filenames = load_features()

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img, model):
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    return result / norm(result)

st.title("Image Similarity Search")
st.header("Upload an image to find similar images")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)

        img_resized = img.resize((224, 224))
        query_features = extract_features(img_resized, model)

        num_neighbors = st.slider("Number of similar images to display:", 1, 20, 10)
        neigh = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute', metric='euclidean')
        neigh.fit(feature_list)

        distances, indices = neigh.kneighbors([query_features])

        st.header("Similar Images")
        for index in indices[0]:
            similar_img_path = filenames[index]
            try:
                similar_img = Image.open(similar_img_path).convert('RGB')
                st.image(similar_img, caption=f'Similar Image: {os.path.basename(similar_img_path)}', use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load image {os.path.basename(similar_img_path)}: {e}")
    except Exception:
        st.error("The uploaded file is not a valid image.")
