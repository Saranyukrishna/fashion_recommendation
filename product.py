from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocess_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocess_img).flatten()
        final_result = result / norm(result)
        return final_result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


if os.path.exists('filenames.pkl') and os.path.exists('embeddings.pkl'):
    filenames = pickle.load(open('files.pkl', 'rb'))
    feature_list = pickle.load(open('list.pkl', 'rb'))
    start_index = len(feature_list)
else:
    filenames = []
    feature_list = []
    start_index = 0

image_dir = 'images'

valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
all_filenames = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                 os.path.splitext(f)[1].lower() in valid_extensions]

filenames_to_process = all_filenames[start_index:]

for file in tqdm(filenames_to_process):
    feature = features(file, model)
    if feature is not None:
        feature_list.append(feature)
        filenames.append(file)

pickle.dump(feature_list, open('list.pkl', 'wb'))
pickle.dump(filenames, open('files.pkl', 'wb'))
