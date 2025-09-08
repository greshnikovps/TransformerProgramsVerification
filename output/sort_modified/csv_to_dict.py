import pandas as pd

# Загрузка весов
df = pd.read_csv("sort_weights.csv", index_col=[0, 1])

# Преобразование в словарь
weights_dict = {
    feature: {
        key: list(map(float, df.loc[(feature, key)].values))
        for key in df.loc[feature].index
    }
    for feature in df.index.levels[0]
}

import json

with open("weights_hardcoded.py", "w") as f:
    f.write("classifier_weights_dict = ")
    json.dump(weights_dict, f, indent=4)