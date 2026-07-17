import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. Load the Data
script_dir = Path(__file__).resolve().parent
data_file = script_dir / "drawndata1.csv"

# Optional: Add a check to ensure the file exists so it doesn't crash silently
if not data_file.exists():
    print(f"Error: Could not find {data_file}")
else:
    df = pd.read_csv(data_file)

    # 2. Prepare features (X) and target (y)
    X = df[['x', 'y']].values
    y = df['z'] == "a"

    # Create a grid of points for plotting the decision boundary background
    X_new = np.concatenate([
        np.random.uniform(0, X[:, 0].max(), (5000, 1)),
        np.random.uniform(0, X[:, 1].max(), (5000, 1))
    ], axis=1)

    # 3. Define the modeling pipeline function
    def plot_output(scaler):
        pipe = Pipeline([
            ("scale", scaler),
            ("model", KNeighborsClassifier(n_neighbors=20, weights="distance"))
        ])
        return pipe

    # 4. Define the scalers to compare
    scalers = [
        ("Original Data", None), 
        ("StandardScaler", StandardScaler()),
        ("QuantileTransformer", QuantileTransformer(n_quantiles=100))
    ]

    # 5. Plotting Loop
    plt.figure(figsize=(15, 4))

    for i, (name, scaler) in enumerate(scalers):
        plt.subplot(1, 3, i + 1)
        
        if scaler is None:
            # Plotting just the original raw data
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor='k')
        else:
            # Create pipeline, fit the model, and predict the background grid
            pipe = plot_output(scaler)
            pipe.fit(X, y)
            
            # Predict probabilities for the background grid
            y_proba = pipe.predict_proba(X_new)
            
            # Plot the background predictions (light) and the actual data (dark)
            plt.scatter(X_new[:, 0], X_new[:, 1], c=y_proba[:, 1], alpha=0.3)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolor='k')
            
        plt.title(name)

    plt.tight_layout()
    plt.show()