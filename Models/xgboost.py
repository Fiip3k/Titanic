### RUN CleanData.py

# %% Split data for validation
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

# %% Create XGB model
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=1000, learning_rate=0.01, n_jobs=6, eval_metric="logloss")

print("Fitting the model.")
model.fit(train_X, train_y, 
            early_stopping_rounds=10, 
            eval_set=[(test_X, test_y)], 
            verbose=True)

predictions = model.predict(test_X)
predictions.round()
print("Predictions ready.")

# %% Evaluate XGB model

from sklearn.metrics import accuracy_score
error = accuracy_score(test_y, predictions)
error
# %%
