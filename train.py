# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Activation, Bidirectional, TimeDistributed
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint


from sklearn.model_selection import train_test_split


from dataset import *
from process import gen_ngrams_set


import config
import pickle


NGRAM = config.NGRAM
MAXLEN = config.MAXLEN
VALIDATION_SIZE = config.VALIDATION_SIZE
TRAIN_SIZE = config.TRAIN_SIZE
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
HIDDEN_UNITS = config.HIDDEN_UNITS


# def create_model(maxlen, vocab_size, units=256):
#     model = Sequential()
#     model.add(LSTM(units, input_shape=(maxlen, vocab_size), return_sequences=True))
#     model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)))
#     model.add(TimeDistributed(Dense(128)))
#     model.add(Dense(vocab_size))
#     model.add(Activation("softmaxt"))
#     model.compile(
#         loss="categorical_cross_entropy",
#         optimizer=Adam(lr=0.001),
#         metrics=["accuracy"]
#     )
#     model.summary()
#     return model

import pickle

with open("idxabc.pickle", "rb") as f:
    alphabet = dict(pickle.load(f))

print(alphabet)

# models = create_model(maxlen=MAXLEN, vocab_size=len(alphabet), units=HIDDEN_UNITS)

data = load_data("data/test.xlsx")

ngrams = gen_ngrams_set(data, maxlen=MAXLEN, ngr=NGRAM)
print(ngrams[:5])

# if (TRAIN_SIZE + VALIDATION_SIZE) > len(ngrams):
#     train_set, validation_set = train_test_split(ngrams, test_size=0.2)
# else:
#     train_set = ngrams[:TRAIN_SIZE]
#     validation_set = ngrams[TRAIN_SIZE:(TRAIN_SIZE + VALIDATION_SIZE)]

# print("\tTrain size: {}, validation_size: {}".format(len(train_set), len(validation_set)))

# train_generator = generator(train_set, batch_size=BATCH_SIZE, n=NGRAM)
# validation_generator = generator(validation_set, batch_size=BATCH_SIZE, n=NGRAM)

# checkpoint = ModelCheckpoint("best_model.hdf5", verbose=1, save_best_only=true, mode="auto")

# history = models.fit_genorator(train_generator, steps_per_epoch=len(train_set) // BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[checkpoint], validation_data=validation_generator, validation_steps=len(validation_set) // BATCH_SIZE)

# model_json = models.to_json()
# with open("model.json", "wb") as json_file:
#     json_file.write(model_json)


# from matplotlib import pyplot as plt

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

