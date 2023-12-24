import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import load_model
from preprocess import load_food_review_data
from model import get_model
from global_variables import *
    
X_train, X_validation, X_test, y_train, y_validation, y_test, vocab = load_food_review_data()

#increased vocab size for test
vocab_size = len(vocab) +1

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=SEQUENCE_LENGTH)
X_validation = sequence.pad_sequences(X_validation, maxlen=SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=SEQUENCE_LENGTH)

y_train = y_train.astype(np.int32)
y_validation = y_validation.astype(np.int32)
y_test = y_test.astype(np.int32)

def train_model():
    model = get_model(vocab_size, sequence_length=SEQUENCE_LENGTH, embedding_size=EMBEDDING_SIZE, num_classes=NUM_CLASSES)
    # Compiler le modèle
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

    checkpointer = ModelCheckpoint(
        PRETRAINED_MODEL_PATH, save_best_only=True, verbose=1)

    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(
        X_validation, y_validation), batch_size=BATCH_SIZE, callbacks=[checkpointer])

train_model()

model = load_model(PRETRAINED_MODEL_PATH)

# Evaluate the model on the test data
loss, acc = model.evaluate(X_test, y_test)
print(f'Accuracy: {acc:.4f}')
print(f'Loss: {loss:.4f}')