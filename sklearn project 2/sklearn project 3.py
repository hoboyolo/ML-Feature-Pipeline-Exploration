from multiprocessing import Pipe

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path

# Define the correct path
script_dir = Path(__file__).resolve().parent
data_file = script_dir / "drawndata2.csv"

# Use the correct path to load the data
df = pd.read_csv(data_file)

# Use the dataframe
print(df.head(3))

X = df [['x', 'y']].values
y = df['z'] == "a"

plt.scatter (X[:, 0], X[:, 1], c=y)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline 

pipe = Pipeline([
    ("scale", PolynomialFeatures()),
    ("model", LogisticRegression())
])

pred = pipe.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred)

arr = np.array(["low", "low", "high", "medium"]).reshape(-1, 1)
#print (arr)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
enc.fit_transform(arr)
enc.transform([["zero"]])

from sklearn.metrics import classification_report, confusion_matrix

pred = pipe.predict(X)
print(classification_report(y, pred))
print(confusion_matrix(y, pred))

