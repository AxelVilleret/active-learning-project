import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.optimizers import Adam
from preprocess import load_food_review_data
from model import get_model
from config import sequence_length, embedding_size, batch_size, epochs, num_classes
    
X_train, X_test, y_train, y_test, vocab = load_food_review_data()

#vocab_size = len(vocab)

#increased vocab size for test
vocab_size = len(vocab) +1

print("Vocab size:", vocab_size)

X_train = sequence.pad_sequences(X_train, maxlen=sequence_length)
X_test = sequence.pad_sequences(X_test, maxlen=sequence_length)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)



print("X_train.shape:", X_train.dtype)
print("X_test.shape:", X_test.dtype)

print("y_train.shape:", y_train.dtype)
print("y_test.shape:", y_test.dtype)
 
model = get_model(vocab_size, sequence_length=sequence_length, embedding_size=embedding_size, num_classes=num_classes)
# Compiler le modèle
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    "models/base_0.h5", save_best_only=True, verbose=1)

model.fit(X_train, y_train, epochs=epochs, validation_data=(
    X_test, y_test), batch_size=batch_size, callbacks=[checkpointer])

# Sauvegarder le modèle
# model.save('models/base_0.h5')
