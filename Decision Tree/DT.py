import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import pandas as pd
import time

begining_time = time.time()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE



print('\n\n************ Data Parsing ************')


train_filename = '../census/labels.data'

with open(train_filename, 'r', encoding='utf8') as label_file:
    print('Reading label file...')
    train_lines = label_file.readlines()

print('Parsing label file...')
# key -> set of values
column_names = []
column_values = []
for l in train_lines:
    l = l.lower().strip()
    column_index = l.index(':')
    column_names.append(l[:column_index])
    values = l[column_index + 2:]
    if values == 'continuous':
        column_values.append('continuous')
    else:
        values = [x.strip() for x in values.split(',')]
        column_values.append(set(values))

def parse_data_lines(data_line, data_labels, data_weight, data_matrix):
    # -1 to remove the ending period
    splits = l[:-1].lower().split(',')
    trimmed = [v.strip() for v in splits]
    line = []
    for i in range(len(trimmed)):
        v = trimmed[i]

        # Instance weight column
        if i == 24:
            data_weight.append(float(v))
            continue

        # label column
        if i == 41:
            if '-' in v:
                data_labels.append(0)
            else:
                data_labels.append(1)
                continue
        else: 
            if v == '?':
                line.append(None)
                continue
            
            column_value = column_values[i]
            
            if column_value == 'continuous':
                line.append(float(v))
            else:
                if v not in column_value:
                    print('Data not match desired value for column [$d]: ', i, v)
                    line.append(None)
                else:
                    line.append(v)
    data_matrix.append(line)

data_filename = '../census/census-income.data'
test_filename= '../census/census-income.test'

with open(data_filename, 'r', encoding='utf8') as data_file:
    print('Reading train data...')
    data_lines = data_file.readlines()
    
with open(test_filename, 'r', encoding='utf8') as test_file:
    print('Reading test data...')
    test_lines = test_file.readlines()

data_labels = []
data_weight = []
data_matrix = []
print('Parsing train and test data...')
start = time.time()
for l in data_lines:
    parse_data_lines(l, data_labels, data_weight, data_matrix)

test_labels = []
test_weight = []
test_matrix = []
for l in test_lines:
    parse_data_lines(l, test_labels, test_weight, test_matrix)
    
print("Parsing all data lines took: ", time.time()-start)


# Convert arrays into numpy array
data_labels = np.array(data_labels)
data_weight = np.array(data_weight)
data_matrix = np.array(data_matrix)

test_labels = np.array(test_labels)
test_weight = np.array(test_weight)
test_matrix = np.array(test_matrix)

# Remove instance weight column from column name and values
del(column_names[24])
del(column_values[24])


print('\n\n************ Preprocessing ************')
categorical_columns = list(filter(lambda x: column_values[x] != 'continuous', [i for i in range(len(column_values))]))
print("Number of nominal features: ", len(categorical_columns))

categorical_matrix_train = data_matrix[:,categorical_columns]
categorical_matrix_test = test_matrix[:, categorical_columns]
categorical_values = (np.array(column_values))[categorical_columns]
categorical_values = [list(l) for l in categorical_values]

print('Transforming nominal features into binary...')
# Ignore is needed to neglect 'Other' type with question marks
ohe = OneHotEncoder(categories=categorical_values, handle_unknown='ignore')
ohe.fit(categorical_matrix_train)

categorical_sparse_matrix_train = ohe.transform(categorical_matrix_train)
categorical_sparse_matrix_test = ohe.transform(categorical_matrix_test)


continuous_columns = list(filter(lambda x: column_values[x] == 'continuous', [i for i in range(len(column_values))]))

continuous_dense_matrix_train = np.array(data_matrix[:, continuous_columns], dtype=float)
continuous_dense_matrix_test = np.array(test_matrix[:, continuous_columns], dtype=float)

continuous_sparse_matrix_train = scipy.sparse.csr_matrix(continuous_dense_matrix_train)
continuous_sparse_matrix_test = scipy.sparse.csr_matrix(continuous_dense_matrix_test)

print('Combining sparse matrices for nominal and continuous features...')
sparse_data_matrix = scipy.sparse.hstack([continuous_dense_matrix_train, categorical_sparse_matrix_train], format='csr')
sparse_test_matrix = scipy.sparse.hstack([continuous_dense_matrix_test, categorical_sparse_matrix_test], format='csr')

print("Train matrix shape & nnz: ", sparse_data_matrix.shape, sparse_data_matrix.nnz)
print("Test matrix shape & nnz: ", sparse_test_matrix.shape, sparse_test_matrix.nnz)


def get_measures(expected, actual, average='binary'):
    accuracy = accuracy_score(expected, actual)
    precision, recall, f_score, s = precision_recall_fscore_support(expected, actual, average=average)
    return accuracy, precision, recall, f_score

def run_dt_with_params(
    train_weights=[],
    criterion='gini', 
    max_depth=None,
    min_impurity_decrease=0.0,
    boost=False,
    feature_selections=None, 
    average='binary'
):
    train_measures = np.zeros(6, dtype=float)
    test_measures = np.zeros(6, dtype=float)
    depth = 0
    dt = DecisionTreeClassifier(
        criterion=criterion, 
        max_depth=max_depth, 
        min_impurity_decrease=min_impurity_decrease
    )

    if boost:
        gbc = GradientBoostingClassifier(
            init=dt, 
            max_depth=max_depth, 
            min_impurity_decrease=min_impurity_decrease
        )
        dt = gbc

    if feature_selections:
        d_matrix = sparse_data_matrix[:,feature_selections]
        t_matrix = sparse_test_matrix[:,feature_selections]
    else:
        d_matrix = sparse_data_matrix
        t_matrix = sparse_test_matrix
        
    start = time.time()
    if len(train_weights) > 0:
        dt.fit(d_matrix, data_labels, sample_weight=train_weights)
    else:
        dt.fit(d_matrix, data_labels)

    training_time = time.time()-start

    start = time.time()
    train_results = dt.predict(d_matrix)
    test_results = dt.predict(t_matrix)
    predicting_time = time.time()-start

    train_measures += list(get_measures(data_labels, train_results, average=average) + (training_time, predicting_time))
    test_measures += list(get_measures(test_labels, test_results, average=average) + (training_time, predicting_time))
    if not boost:
        depth += dt.tree_.max_depth
    
    return train_measures, test_measures, depth
    

def plot_measures(x_axis, measures, x_label, runtime=True):
    plt.plot(x_axis, measures[:, 0:4], label=['accuracy','precision','recall','f-score'])
    plt.xlabel(x_label)
    plt.title('Measures vs ' + x_label)
    plt.legend()
    plt.show()

    if runtime:
        plt.plot(x_axis, measures[:, 4:], label=['Training time', 'Predicting time'])
        plt.xlabel(x_label)
        plt.ylabel('seconds')
        plt.title('Runtime vs ' + x_label)
        plt.legend()
        plt.show()


print('\n\n************ Training and Prediction ************')
print('Running Decision Tree with min_impurity_decrease from 1*10^-6 to 5*10^-5')

x = [i/(10**6) for i in range(50, 0, -1)]
dt_impurity_instance_train = []
dt_impurity_instance_test = []
dt_impurity_instance_depths = []
for depth in x:
    a,b,c = run_dt_with_params(
        train_weights=data_weight,
        min_impurity_decrease=depth, 
        average='weighted')
    dt_impurity_instance_train.append(a)
    dt_impurity_instance_test.append(b)
    dt_impurity_instance_depths.append(c)

dt_impurity_instance_train = np.array(dt_impurity_instance_train)
dt_impurity_instance_test = np.array(dt_impurity_instance_test)
dt_impurity_instance_depths = np.array(dt_impurity_instance_depths)

plot_measures(x, dt_impurity_instance_train, 'Min Impurity Decrease (Train)')
plot_measures(x, dt_impurity_instance_test, 'Min Impurity Decrease (Test)', runtime=False)


plt.plot(x, dt_impurity_instance_depths, label=['Max depth'])
plt.ylabel('Max Depth')
plt.xlabel('Min Impurity Decrease')
plt.title('Max Depth vs Min Impurity Decrease')
plt.show()


max_fscore_index = np.argmax(dt_impurity_instance_test[:,3])
best_measure = dt_impurity_instance_test[max_fscore_index]
print("Result from the best f-score")
print("Minimum impurity decrease: ", x[max_fscore_index])
print("Measures (accuracy, precision, recall, f-score): (%f, %f, %f, %f)" % (best_measure[0], best_measure[1], best_measure[2], best_measure[3]))
print('Maximum depth: ', dt_impurity_instance_depths[max_fscore_index])
np.sum(dt_impurity_instance_test[:, 4])




print('\n\n************ End of Decision Tree ************')
print('Total run time: %f seconds' % (time.time()-begining_time))






























        