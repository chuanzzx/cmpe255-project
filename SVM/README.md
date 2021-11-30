# SVM training/testing

## Prerequisites
Need to include the original dataset and "labels.data" at the parent folder.
```
../data/census-income.data
../data/census-income.test
../data/labels.data
```

### Required Python Packages
```
numpy
joblib
sklearn
```

## Usage
Training: run the following command
```
python SVM_train.py
```

Testing: run the following command with your model path
```
python SVM_test.py [model path]
python SVM_test.py ./model/svm_model_c3.joblib
```