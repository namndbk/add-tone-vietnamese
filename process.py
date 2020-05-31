from nltk import ngrams
from random import shuffle
import re
import itertools


def extract_phrases(text):
    text = text.lower().strip()
    return re.findall(r'\w[\w ]+', text, re.UNICODE)


def gen_ngram_from_text(text, n):
    words = text.split()
    if len(words) < n:
        return ngrams(words, len(words))
    return ngrams(words, n)


def gen_ngram_set(corpus, maxlen=32, ngr=5):
    list_ngram = []
    phrases = itertools.chain.from_iterable(extract_phrases(text) for text in corpus)
    phrases = [p for p in phrases if len(p.split()) >= 2]
    for phrase in phrases:
        for ngram in gen_ngram_from_text(phrase, ngr):
            sent = " ".join(token for token in ngram)
            if len(sent) <= maxlen:
                n = len(sent)
                sent += "\x00" * (maxlen - n)
                list_ngram.append(sent)
    del phrases
    list_ngram = list(set(list_ngram))
    print(len(list_ngram))
    shuffle(list_ngram)
    return list_ngram
