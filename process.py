from nltk import ngrams
from random import shuffle

import unidecode
import re
import numpy as np
import itertools


def extract_phrases(text):
    text = text.lower().strip()
    return re.findall(r'\w[\w ]+', text, re.UNICODE)


def gen_ngrams_from_text(text, n):
    return ngrams(text.split(), n)


def gen_ngrams_set(corpus, maxlen=32, ngr=5):
    list_ngrams = []
    phrases = itertools.chain.from_iterable(extract_phrases(text) for text in corpus)
    phrases = [p for p in phrases if len(p.split()) >= 2]
    print(len(phrases))
    for phrase in phrases:
        for ngram in gen_ngrams_from_text(phrase, ngr):
            sent = " ".join(token for token in ngram)
            if len(sent) <= maxlen:
                n = len(sent)
                sent += "\x00" * (maxlen - n)
                list_ngrams.append(sent)
    del phrases
    list_ngrams = list(set(list_ngrams))
    print(len(list_ngrams))
    shuffle(list_ngrams)
    return list_ngrams