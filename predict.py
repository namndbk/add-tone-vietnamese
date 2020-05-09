from keras.models import Sequential
from keras.models import model_from_json

try:
    json_file = open("model.json", "r")
    model_json = json_file.read()
    models = model_from_json(model_json)
    models.load_weights("best_model.hdf5")
except Exception as e:
    raise e
