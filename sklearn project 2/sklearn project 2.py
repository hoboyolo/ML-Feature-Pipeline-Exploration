from multiprocessing import Pipe

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path

# Define the correct path
script_dir = Path(__file__).resolve().parent
data_file = script_dir / "drawndata1.csv"

# Use the correct path to load the data
df = pd.read_csv(data_file)

# Use the dataframe
print(df.head(3))

X = df [['x', 'y']].values
y = df['z'] == "a"

plt.scatter (X[:, 0], X[:, 1], c=y)

from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)

def plot_output(scaler):
    pipe = Pipeline([
        ("scale", scaler),
        ("model", KNeighborsClassifier(n_neighbors=20, weights="distance"))
    ])
    return pipe

scaler = StandardScaler()
pipe = plot_output(scaler)
pred = pipe.fit(X, y).predict(X)

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Original Data")
plt.subplot(132)
X_tfm = scaler.transform(X)
plt.scatter(X_tfm[:, 0], X_tfm[:, 1], c=y)
plt.title("Transformed Data")
plt.subplot(133)
X_new = np.concatenate([
    np.random.uniform(0, X[:, 0].max(), (5000, 1)),
    np.random.uniform(0, X[:, 1].max(), (5000, 1))
], axis =1)
y_proba = pipe.predict_proba(X_new)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y_proba[:, 1], alpha=0.7)
plt.title("Predicted Data")
plot_output(scaler=StandardScaler())
plot_output(scaler=QuantileTransformer(n_quantiles=100))
plt.tight_layout()
plt.show()



