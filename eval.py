from keras.models import load_model
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from preprocess import load_food_review_data, load_cloth_review_data
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import sequence
from config import sequence_length, embedding_size, batch_size, epochs

# Replace 'pretrained_model_path' with the actual path to your pretrained model.
pretrained_model_path = 'models/base_0.h5'
pretrained_model = load_model(pretrained_model_path)

# X_train, X_test, y_train, y_test, vocab = load_cloth_review_data()
X_train, X_test, y_train, y_test, vocab = load_food_review_data()

X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

#y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.int32)

# Evaluate the model on the test data
y_pred = pretrained_model.predict(X_test)

# print(y_test.shape)

acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))
print(f'Accuracy: {acc:.4f}')

