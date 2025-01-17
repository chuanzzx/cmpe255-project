import numpy as np
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

    # "GRINST" 22, "MARSUPWT" 25, 
    # "MIGMTR1" 26, "MIGMTR3" 27, "MIGMTR4" 28, "MIGSUN" 30, 
    # "PEFNTVTY" 33, "PEMNTVTY" 34, "PENATVTY" 35, "YEAR" 41
    drop_cols = set([21, 24, 25, 26, 27, 29, 32, 33, 34, 40])

    with open(path, "r", encoding="utf8") as f:
        lines = f.readlines()
        
        for line in lines:
            raw_data = line.split(",")
            raw_X = [x.strip() for x in raw_data[:-1]]
            out_y.append(raw_data[-1])

            # one data features
            X = []
            for i, x in enumerate(raw_X):
                # if i in drop_cols:
                #     # not select this attribute
                #     continue

                if i == 24:
                    # skip weight instance
                    continue

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

# sampling total_num data (50% class0, 50% class1)
def sample_balanced_data(X, y, total_num):
    num = total_num // 2
    idx_class0 = np.where(y == " - 50000.\n")[0]
    idx_class1 = np.where(y == " 50000+.\n")[0]
    pick = np.concatenate((np.random.choice(idx_class0, num), np.random.choice(idx_class1, num)))
    return X[pick], y[pick]
    
def main():
    # prepare for categorical encoding
    column_file = "../data/labels.data"
    column_values = parse_column_types(column_file)

    # parse training data
    train_path = "../data/census-income.data"
    train_X, train_y = read_data(train_path, column_values)
    print (train_X.shape, train_y.shape)

    # parse testing data
    test_path = "../data/census-income.test"
    test_X, test_y = read_data(test_path, column_values)
    print (test_X.shape, test_y.shape)

    # SVM model
    print ("Building SVM model...")
    clf = SVC(C=3) # C, default=1.0

    print ("Fiting...")
    clf.fit(train_X, train_y)

    # save the model
    print("Saving model...")
    dump(clf, "./models/test.joblib") 

    # predict
    print ("Predicting...")
    pred_y = clf.predict(test_X[:])

    # evaluate predicted results
    print ("\nf1_score weighted")
    # print ("f1_score None/micro/macro")
    # print (f1_score(test_y, pred_y, average=None))
    # print (f1_score(test_y, pred_y, average="micro"))
    # print (f1_score(test_y, pred_y, average="macro"))
    print (f1_score(test_y, pred_y, average="weighted"))
    # print (precision_recall_fscore_support(test_y, pred_y))

if __name__ == '__main__':
    main()