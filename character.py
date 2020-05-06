import numpy as np

import utils
import json


class CharacterModel(object):

	def __init__(self, index_alphabet, maxlen=32):
		self.index_alphabet = index_alphabet
		self.maxlen = maxlen
	
	def encode(self, text, maxlen):
		try:
			x = np.zeros((self.maxlen, len(self.alphabet)))
			for i, c in enumerate(text[:self.maxlen]):
				x[i, self.index_alphabet[c]] = 1
			if i < self.maxlen - 1:
				for j in range(i + 1, self.maxlen):
					x[j, self.index_alphabet["$"]] = 1
			return x
		except KeyError:
			return None
	
	def decode(self, x):
		x = np.argmax(x, axis=-1)
		out = " ".join(self.index_alphabet[i] for i in x)
		return out