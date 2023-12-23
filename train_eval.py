from re import L
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import load_model
from preprocess import load_food_review_data
from model import get_model
from config import sequence_length, embedding_size, batch_size, epochs, num_classes
from sklearn.metrics import accuracy_score

PRETRAINED_MODEL_PATH = 'models/base_0.h5'
    
X_train, X_validation, X_test, y_train, y_validation, y_test, vocab = load_food_review_data()

#increased vocab size for test
vocab_size = len(vocab) +1

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
X_validation = sequence.pad_sequences(X_validation, maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

y_train = y_train.astype(np.int32)
y_validation = y_validation.astype(np.int32)
y_test = y_test.astype(np.int32)

def train_model():
    model = get_model(vocab_size, sequence_length=sequence_length, embedding_size=embedding_size, num_classes=num_classes)
    # Compiler le modèle
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

    checkpointer = ModelCheckpoint(
        PRETRAINED_MODEL_PATH, save_best_only=True, verbose=1)

    model.fit(X_train, y_train, epochs=epochs, validation_data=(
        X_validation, y_validation), batch_size=batch_size, callbacks=[checkpointer])

train_model()

model = load_model(PRETRAINED_MODEL_PATH)

# Evaluate the model on the test data
# y_pred = model.predict(X_test)

# acc = accuracy_score(y_test, np.argmax(y_pred, axis=1))

loss, acc = model.evaluate(X_test, y_test)
print(f'Accuracy: {acc:.4f}')
print(f'Loss: {loss:.4f}')