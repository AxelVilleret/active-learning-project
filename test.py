from keras.models import load_model
import numpy as np
from utils import clean_text, tokenize_words
from config import embedding_size, sequence_length
from keras.preprocessing.sequence import pad_sequences
from preprocess import get_dict

vocab2int = get_dict()
model = load_model("models/base_0.h5")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Food Review evaluator")
    parser.add_argument("review", type=str, help="The review of the product in text")
    args = parser.parse_args()

    review = tokenize_words(clean_text(args.review), vocab2int)
    x = pad_sequences([review], maxlen=sequence_length)
    print(f"{np.argmax(model.predict(x))}/5")
