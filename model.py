from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, LeakyReLU, Dropout, TimeDistributed
from keras.layers import SpatialDropout1D
from global_variables import *

def get_model(vocab_size, sequence_length, embedding_size, num_classes, verbose=0):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
              input_length=sequence_length))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(LSTM_UNITS, recurrent_dropout=0.2))
    model.add(Dropout(0.3))
    # Modifié pour la classification multiclasse
    model.add(Dense(num_classes, activation="softmax"))
    if verbose:
        model.summary()
    return model
