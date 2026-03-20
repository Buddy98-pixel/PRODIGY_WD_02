import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize

# 1. LOAD AND PREPROCESS DATA
def load_images(path, label):
    images = []
    labels = []
    for img_name in os.listdir(path)[:500]: # Loading 500 of each for speed
        try:
            img = imread(os.path.join(path, img_name))
            img_resized = resize(img, (64, 64)).flatten() # Flatten 2D image to 1D array
            images.append(img_resized)
            labels.append(label)
        except:
            continue
    return images, labels

# Assume you have folders 'cats' and 'dogs'
# cats_img, cats_lbl = load_images('data/cats', 0)
# dogs_img, dogs_lbl = load_images('data/dogs', 1)

# X = np.array(cats_img + dogs_img)
# y = np.array(cats_lbl + dogs_lbl)

# 2. TRAIN SVM MODEL
# Using RBF kernel is standard for non-linear image data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1.0) # Linear is faster for high-dimensional data
svm_model.fit(X_train, y_train)

# 3. EVALUATE
y_pred = svm_model.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))