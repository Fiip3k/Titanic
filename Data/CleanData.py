# %% Read data from files

import pandas as pd
test_data = pd.read_csv('CSV\\test.csv', index_col=0)
test_full = test_data.copy()
train_data = pd.read_csv('CSV\\train.csv', index_col=0)
train_full = train_data.copy()
print("Data loaded.")

# %% Do bad things to data

X = train_full.copy()
y = X.pop('Survived')
print("Split happens.")

# %% Show data

print("Showing data:")
X

# %% Drop "Name" column

X = X.drop("Name", axis = 1)
test_full = test_full.drop("Name", axis = 1)
print("\"Name\" column dropped.")


# %% LE "Sex" column (probably should OHE)

from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#X["Sex"] = le.fit_transform(X["Sex"])
#test_full["Sex"] = le.transform(test_full["Sex"])


# %% Fix NaN in "Age" column (Simple Impute -> terrible decision xD) 
print("Making a terrible decision.")

X["Age"].fillna(X["Age"].mean(), inplace=True)
test_full["Age"].fillna(test_full["Age"].mean(), inplace=True)

print("Mistake finished. Thank you for your patience.")

# %% Drop "Ticket" column

X = X.drop("Ticket", axis = 1)
test_full = test_full.drop("Ticket", axis = 1)
print("\"Ticket\" column dropped.")

# %% Fare is cool

X["Fare"].isna().sum()
print("\"Fare\" is cool.")
test_full["Fare"].fillna(test_full["Fare"].mean(), inplace = True)

# %% Create "Deck" feature from "Cabin" column

le = LabelEncoder()
X["Deck"] = X["Cabin"].str[0]
#X["Deck"] = le.fit_transform(X["Deck"])
test_full["Deck"] = test_full["Cabin"].str[0]
#test_full["Deck"] = le.transform(test_full["Deck"])

print("\"Deck\" feature created.")

# %% Drop "Cabin" column
X = X.drop("Cabin", axis = 1)
test_full = test_full.drop("Cabin", axis = 1)

# %% Drop "Embarked" column
X = X.drop("Embarked", axis = 1)
test_full = test_full.drop("Embarked", axis = 1)

# %% OHE remaining object columns

from sklearn.preprocessing import OneHotEncoder

object_cols = ["Sex", "Deck"]
ohe = OneHotEncoder(handle_unknown = "ignore", sparse = False)
OH_data = pd.DataFrame(ohe.fit_transform(X[object_cols]))
OH_data.columns = ohe.get_feature_names(object_cols)
OH_data.index = X.index

X = X.drop(object_cols, axis=1)
X = pd.concat([X, OH_data], axis=1)

# %%
X

# %% Normalize columns

from sklearn.preprocessing import MinMaxScaler
column_names = X.columns
numpy_data = X.values
scaler = MinMaxScaler()
numpy_data = scaler.fit_transform(numpy_data)
X = pd.DataFrame(numpy_data)
X.columns = column_names

print("Data normalized.")
X
# %%
