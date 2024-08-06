import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import bentoml

# Load the Iris dataset
iris = load_iris()
X_train = iris.data
y_train = iris.target

# Create and train your scikit-learn model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the scikit-learn model using BentoML
saved_model = bentoml.sklearn.save_model(name="iris_clf", model=model)
print(f"Model saved: {saved_model}")


# added a single line of code