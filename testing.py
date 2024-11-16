import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('list.pkl', 'rb')))
filenames = pickle.load(open('files.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('test/shirts.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocess_img = preprocess_input(expanded_img_array)

result = model.predict(preprocess_img).flatten()
final_result = result / norm(result)

neigh = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neigh.fit(feature_list)

distances, indices = neigh.kneighbors([final_result])

seen_images = set()

for file in indices[0][1:6]:
    if file not in seen_images:
        seen_images.add(file)
        temp_img = cv2.imread(filenames[file])
        cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
        cv2.waitKey(0)

cv2.destroyAllWindows()
