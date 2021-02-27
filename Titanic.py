### RUN CleanData.py

# %% Train full and save results
model = RandomForestClassifier(n_estimators = 800, max_depth = 4)
model.fit(X, y)
predictions = model.predict(test_full)

predictions = predictions.round()
print("Predictions ready.")

output = pd.DataFrame({"PassengerId": test_data.index,
                        "Survived": predictions.astype("int64")})
                        
output.to_csv("submission.csv", index=False)


# %%
predictions
# %%
