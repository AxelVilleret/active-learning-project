import numpy as np
from string import punctuation

# make it a set to accelerate tests
punc = set(punctuation)

#def clean_text(text):
#    return ''.join([ c.lower() for c in text if c not in punc ])


def clean_text(text):
    cleaned_chars = [c.lower() for c in text if c not in punc]
    cleaned_text = ''.join(cleaned_chars)
    return cleaned_text

def tokenize_words(words, vocab2int):
    if isinstance(words, int):
        words = str(words)  # Convert integer to string
    words = words.split()
    tokenized_words = np.zeros((len(words),))
    for j in range(len(words)):
        try:
            tokenized_words[j] = vocab2int[words[j]]
        except KeyError:
            # didn't add any unk, just ignore
            pass
    return tokenized_words
