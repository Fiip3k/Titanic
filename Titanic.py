### RUN CleanData.py

# %% Train full and save results
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=240, learning_rate=0.01, n_jobs=6, eval_metric="logloss")

print("Fitting the model.")
model.fit(XR, yR, verbose=False)

predictions = model.predict(test_full)

print("Predictions ready.")

submission_data = pd.DataFrame({"PassengerId": test_data.index, "Survived": predictions})
submission_data.to_csv("CSV\submission.csv", index = False)

"""
# %%
submission_data
# %%
test_full
# %% Train full and save results OLD
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 240, max_depth = 4)
model.fit(XR, yR)
predictions = model.predict(test_full)

predictions = predictions.round()
print("Predictions ready.")

output = pd.DataFrame({"PassengerId": test_data.index,
                        "Survived": predictions.astype("int64")})
                        
output.to_csv("CSV\submission.csv", index=False)

# %%
"""