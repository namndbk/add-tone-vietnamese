from keras.models import Sequential
from keras.models import model_from_json
from character import CharacterModel
import numpy as np
from preprocess import remove_accent
from collections import Counter
import re

import pickle

from process import gen_ngram_from_text

import string


def extract_phrases(text):
    punction = []
    phrases = []
    index = 0
    for i, c in enumerate(text):
        if c in string.punctuation:
            phrases.append(text[index: i])
            index = i + 1
            punction.append(c)
    return phrases, punction


with open("idxabc.pickle", "rb") as f:
    index_alphabet = dict(pickle.load(f))

try:
    json_file = open("models/modelv2/model_v2.json", "r")
    model_json = json_file.read()
    models = model_from_json(model_json)
    models.load_weights("models/modelv2/best_model_v2.hdf5")
except Exception as e:
    raise e

codec = CharacterModel(index_alphabet=index_alphabet)
print(codec.index_alphabet)


def guess(ngram):
    text = ' '.join(ngram)
    n = len(text)
    if n < 32:
        text += "\x00" * (32 - n)
    preds = models.predict(
        np.array([codec.encode(remove_accent(text))]), verbose=0)
    return codec.decode(preds[0]).strip('\x00')


def add_accent(text):
    ngrams = list(gen_ngram_from_text(text.lower(), n=5))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(text.split()))]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    try:
        output = " ".join(c.most_common(1)[0][0] for c in candidates)
    except:
        output = ""
    print(candidates)
    return output


def accent_sentence(sentence):
    list_phrases, punction = extract_phrases(sentence)
    output = ""
    for i, phrases in enumerate(list_phrases):
        if len(phrases.split()) < 2:
            output += phrases + punction[i] + " "
        else:
            out = add_accent(phrases.lower()).strip()
            for j, c in enumerate(phrases.strip()):
                if c.isupper():
                    output += out[j].upper()
                else:
                    output += out[j]
            output += punction[i] + " "
    print(output)


text = """Thí sinh chỉ được điều chỉnh đăng ký xét tuyển một lần và chỉ được sử dụng một trong hai phương thức trực \
tuyến hoặc bằng phiếu. Với điều chỉnh bằng phương thức trực tuyến, các em sử dụng tài khoản và mật khẩu cá nhân đã \
được cấp. Phương thức này chỉ chấp nhận khi số lượng nguyện vọng sau khi điều chỉnh không lớn hơn số đã đăng ký ban \
đầu. """
text2 = '''Trung Quoc da mo rong anh huong cua ho trong khu vuc thong qua cac buoc leo thang ep buoc cac nuoc lang \
gieng o Hoa Dong, Bien Dong, boi dap dao nhan tao va quan su hoa cac cau truc dia ly tren Bien Dong trai luat phap \
quoc te; Tim cach chia re Hoa Ky khoi cac dong minh chau A thong qua cac no luc ep buoc va leo lai kinh te '''
text3 = "cái nha rung lac vi anh em ba con nhay ram ram nhu muon sap."
accent_sentence(text2)
