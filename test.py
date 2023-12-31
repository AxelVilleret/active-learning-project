from keras.models import load_model
import numpy as np
from utils import clean_text, tokenize_words
from keras.preprocessing.sequence import pad_sequences
from preprocess import get_dict
from global_variables import *

vocab2int = get_dict()
model = load_model(PRETRAINED_MODEL_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Food Review evaluator")
    parser.add_argument("review", type=str, help="The review of the product in text")
    args = parser.parse_args()

    review = tokenize_words(clean_text(args.review), vocab2int)
    x = pad_sequences([review], maxlen=SEQUENCE_LENGTH)
    print(f"{np.argmax(model.predict(x))}/5")
