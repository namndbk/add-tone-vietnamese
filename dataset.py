import numpy as np
import pandas as pd
import pickle

from preprocess import remove_accent

from character import CharacterModel

from config import MAXLEN


def load_data(file_path):
    df = pd.read_excel(file_path)
    return df.text

with open("idxabc.pickle", "rb") as f:
    index_alphabet = dict(pickle.load(f))


character = CharacterModel(index_alphabet=index_alphabet, maxlen=MAXLEN)

def generator(data, batch_size=128):
    index = 0
    n = len(data)
    while True:
        X, y = [], []
        for item in range(batch_size):
            if character.encode(data[index]) is not None:
                X.append(character.encode(remove_accent(data[index])))
                y.append(character.encode(data[index]))
                index += 1
                if index == n:
                    index = 0
            else:
                continue
        yield np.array(X), np.array(y)
        