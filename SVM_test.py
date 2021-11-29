import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.metrics import f1_score, precision_recall_fscore_support

def parse_column_types(path):
    with open(path, "r", encoding="utf8") as label_file:
        train_lines = label_file.readlines()

    column_values = []
    for l in train_lines:
        l = l.lower().strip()
        column_index = l.index(":")
        values = l[column_index + 2:]
        if values == "continuous":
            column_values.append("continuous")
        else:
            values = [x.strip() for x in values.split(',')]

            # prepare for label encoding
            # assign a value to each type in a specific feature category
            column_values.append({val: i for i, val in enumerate(values)})

    # column category values
    return column_values

def read_data(path, column_values):
    out_X = []
    out_y = []

    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        
        # for line in lines[:3]:
        for line in lines:
            raw_data = line.split(",")
            raw_X = [x.strip() for x in raw_data[:-1]]
            out_y.append(raw_data[-1])

            # one data features
            X = []
            for i, x in enumerate(raw_X):
                if column_values[i] == "continuous":
                    X.append(x)
                    continue

                x = x.lower()
                if x in column_values[i]:
                    # label encoder
                    X.append(column_values[i][x])
                else:
                    # missing data
                    # assign "?" as -1
                    X.append(-1)

            # append this data's features
            out_X.append(X)
    
    return np.array(out_X), np.array(out_y)
    
def main():
    # prepare for categorical encoding
    column_file = "./labels.data"
    column_values = parse_column_types(column_file)

    # parse testing data
    test_path = "./census-income.test"
    test_X, test_y = read_data(test_path, column_values)
    print (test_X.shape, test_y.shape)

    # load SVM model
    model_path = sys.argv[1]
    print ("Loading SVM model:", model_path)
    clf = load(model_path)

    # predict
    print ("Predicting...")
    pred_y = clf.predict(test_X[:])

    # evaluate predicted results
    print ("\nf1_score None/micro/macro")
    print (f1_score(test_y, pred_y, average=None))
    print (f1_score(test_y, pred_y, average="micro"))
    print (f1_score(test_y, pred_y, average="macro"))
    print ("\nprecision_recall_fscore_support None/macro/micro/weighted")
    print (precision_recall_fscore_support(test_y, pred_y)) # default=None
    print (precision_recall_fscore_support(test_y, pred_y, average='macro'))
    print (precision_recall_fscore_support(test_y, pred_y, average='micro'))
    print (precision_recall_fscore_support(test_y, pred_y, average='weighted'))

if __name__ == '__main__':
    main()