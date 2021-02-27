### RUN CleanData.py

# %% Split data for validation
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# %% Create RF model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 800, max_depth = 4)
model.fit(train_X, train_y)
predictions = model.predict(test_X)
predictions = predictions.round()
print("Predictions ready.")

# %% Evaluate RF model

from sklearn.metrics import accuracy_score
error = accuracy_score(test_y, predictions)
error