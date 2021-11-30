import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

INSTANCE_NUM = 24764

# Loading in the census training data
census_df = pd.read_csv("census/census-income.data", header=None)

# Imputing missing values ("?") using most frequent policy
imp_census_df = census_df

for col in census_df:
    missing = census_df[col].astype(str).str.contains("\?", na=False).any()

    if missing:
        col_counts = census_df[col].value_counts()
        
        rep_val = col_counts.index[0].strip() if col_counts.index[0].strip() != "?" else col_counts.index[1].strip()
            
        imp_census_df[col] = census_df[col].str.replace("?", rep_val)

# Rename class column
imp_census_df = imp_census_df.rename(columns={41: "labels"})

# one hot encoding categorical variables to make them numerical
ohe = OneHotEncoder(sparse=False)
cat_feat_df = imp_census_df.select_dtypes("object") # selecting the string type columns
ohe.fit(cat_feat_df.drop(["labels"], axis=1))

# one hot encoder transforming
codes = ohe.transform(cat_feat_df.drop(["labels"], axis=1))
one_hot_df = pd.concat([imp_census_df.select_dtypes(exclude="object"),
                        pd.DataFrame(codes, columns=ohe.get_feature_names()).astype(int)], axis=1)
one_hot_df["labels"] = census_df["labels"]

# splitting data based on label
under_df = one_hot_df.loc[one_hot_df.labels == " - 50000."]
over_df = one_hot_df.loc[one_hot_df.labels == " 50000+."]

# undersample the dominant class
under_idx = under_df.index
random_idx = np.random.choice(under_idx, INSTANCE_NUM, replace=False)
under_samp_df = under_df.loc[random_idx]

# combine minority and majority class instances
comb_df = pd.concat([over_df, under_samp_df])
comb_df = comb_df.rename(columns={41: "labels"})

train_df = comb_df

# SMOTE sampling of minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(train_df.drop(["labels"], axis=1), comb_df["labels"])

train_df = X_resampled
train_df["labels"] = y_resampled

# PCA (scaling before dimensionality reduction)
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_df.drop(["labels"], axis=1))

pca = PCA(n_components=0.99)
pca.fit(train_data_scaled)

# creating training DataFrame
train_red_df = pca.transform(train_data_scaled)
training_df = pd.DataFrame(train_red_df)
training_df["labels"] = np.asarray(train_df["labels"])

# splitting dataset before train_test_split
train_neg = training_df[training_df.labels == " - 50000."]
train_pos = training_df[training_df.labels == " 50000+."]

X_neg = train_neg.drop(["labels"], axis=1)
y_neg = train_neg["labels"]
X_pos = train_pos.drop(["labels"], axis=1)
y_pos = train_pos["labels"]

# train test splitting
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)
X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(X_neg, y_neg, test_size=0.2, random_state=42)

# concatenating training and testing splits
X_train = pd.concat([X_train_pos, X_train_neg])
y_train = pd.concat([y_train_pos, y_train_neg])
X_test = pd.concat([X_test_pos, X_test_neg])
y_test = pd.concat([y_test_pos, y_test_neg])

'''
Grid Search with Random Forset Classifiers. Commented out so that the execution of this script is not too long.

# classifier
rf = RandomForestClassifier()

# parameter lists
n_estimators = [100, 300]
max_depth = [None]
min_samples_split = [2, 5]
min_samples_leaf = [1, 5]

# parameter grid
param_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf
}

rf_grid_clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
rf_grid_clf.fit(X_train, y_train)

The best params after grid search:

{
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 300
}
'''

# train using full training set
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

final_clf = RandomForestClassifier(n_estimators=300, min_samples_split=5, min_samples_leaf=1, max_depth=None)
final_clf.fit(X, y)

# loading test set
census_test_df = pd.read_csv("census/census-income.test", header=None)

# preprocessing on test data
imp_test_df = census_test_df

for col in imp_test_df:
    missing = imp_test_df[col].astype(str).str.contains("\?", na=False).any()
    
    if missing:
        col_counts = imp_test_df[col].value_counts()
        
        rep_val = col_counts.index[0].strip() if col_counts.index[0].strip() != "?" else col_counts.index[1].strip()
            
        imp_test_df[col] = imp_test_df[col].str.replace("?", rep_val)

imp_test_df = imp_test_df.rename(columns={41: "labels"})
        
# one hot encoding
test_df = ohe.transform(imp_test_df.select_dtypes("object").drop(["labels"], axis=1))
test_df = pd.concat([imp_test_df.select_dtypes(exclude="object"),
                     pd.DataFrame(test_df).astype(int)], axis=1)
test_df["labels"] = imp_test_df["labels"]

# PCA
test_data_scaled = scaler.fit_transform(test_df.drop(["labels"], axis=1))

test_red_df = pca.transform(test_data_scaled)
testing_df = pd.DataFrame(test_red_df)
testing_df["labels"] = np.asarray(test_df["labels"])

# predicting
predictions = final_clf.predict(testing_df.drop(["labels"], axis=1))
print(classification_report(testing_df["labels"], predictions))
print("Accuracy of model: " + str(accuracy_score(testing_df["labels"], predictions)))