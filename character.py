import numpy as np


class CharacterModel(object):

	def __init__(self, index_alphabet, maxlen=32):
		self.index_alphabet = index_alphabet
		self.maxlen = maxlen
		self.alphabet_index = {self.index_alphabet[i]: i for i in index_alphabet}
	
	def encode(self, text):
		try:
			x = np.zeros((self.maxlen, len(self.index_alphabet)))
			for i, c in enumerate(text[:self.maxlen]):
				x[i, self.index_alphabet[c]] = 1
			return x
		except Exception as e:
			raise e
			return None
	
	def decode(self, x):
		x = np.argmax(x, axis=-1)
		out = "".join(self.alphabet_index[i] for i in x)
		return out
