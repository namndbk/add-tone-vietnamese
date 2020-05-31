from model import ToneModel
from dataset import load_data
import pandas as pd
import config
import time


model_file = "models/modelv2/model_v2.json"
weights_file = "models/modelv2/best_model_v2.hdf5"
alphabet_file = "idxabc.pickle"
model = ToneModel(config, model_file, weights_file, alphabet_file)
data = load_data("data/train.xlsx")
count = 0
y_true = 0
y_pred = 0
df = pd.DataFrame()
text_true = []
text_pred = []
start_time = time.time()
for sent in data:
    if count == 10:
        break
    else:
        for line in sent.split("\n"):
            line = line.strip()
            y_p, y_p = 0, 0
            if line.strip():
                try:
                    y_p, y_t, out = model.add_tone_v2(line)
                except Exception as e:
                    print(line)
                    print(out)
                    print("=" * 50)
                    out = "None"
                    assert e
                text_pred.append(out)
                text_true.append(line)
                y_pred += y_p
                y_true += y_t
        count += 1
df["text_true"] = text_true
df["text_pred"] = text_pred
test_time = time.time() - start_time
df.to_csv("test_10.csv")
print("\tTest time: %.2f" % test_time)
print("\tAccuracy: %.2f" % (y_pred * 100 / y_true))
