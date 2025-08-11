import json
import math
import numpy as np
import os
import sys
import random

import math

from numpy import genfromtxt
import numpy as np

NUM_USER = 2

training_dir = "raw_data/adult.train"
testing_dir = "raw_data/adult.test"

inputs = (
    ("age", ("continuous",)),
    (
        "workclass",
        (
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ),
    ),
    ("fnlwgt", ("continuous",)),
    (
        "education",
        (
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ),
    ),
    ("education-num", ("continuous",)),
    (
        "marital-status",
        (
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ),
    ),
    (
        "occupation",
        (
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ),
    ),
    (
        "relationship",
        (
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ),
    ),
    ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
    ("sex", ("Female", "Male")),
    ("capital-gain", ("continuous",)),
    ("capital-loss", ("continuous",)),
    ("hours-per-week", ("continuous",)),
    (
        "native-country",
        (
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ),
    ),
)


def isFloat(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False


def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value):
                summ += float(value)
                count += 1
        means.append(summ / count)
    return means


def generate_dataset(file_path):
    input_shape = []
    for i in inputs:
        count = len(i[1])
        input_shape.append(count)
    input_dim = sum(input_shape)

    outputs = (0, 1)  # (">50K", "<=50K")
    output_dim = 2  # len(outputs)

    # input_shape: [1, 8, 1, 16, 1, 7, 14, 6, 5, 2, 1, 1, 1, 41]
    # input_dim: 105 (99)

    # output_dim: 2


def prepare_data(raw_data, means):
    X = raw_data[:, :-1]
    y = raw_data[:, -1:]
    print(y)

    # X:
    def flatten_persons_inputs_for_model(person_inputs, means):
        input_shape = [1, 8, 1, 16, 1, 7, 14, 6, 5, 2, 1, 1, 1, 41]
        float_inputs = []

        for i in range(len(input_shape)):
            features_of_this_type = input_shape[i]
            is_feature_continuous = features_of_this_type == 1

            if is_feature_continuous:
                # in order to be consistant with the google paper -- only train with categorical features
                """
                mean = means[i]
                if isFloat(person_inputs[i]):
                    scale_factor = 1/(2*mean)  # we prefer inputs mainly scaled from -1 to 1.
                    float_inputs.append(float(person_inputs[i])*scale_factor)
                else:
                    float_inputs.append(mean)
                """
                pass
            else:
                for j in range(features_of_this_type):
                    feature_name = inputs[i][1][j]

                    if feature_name == person_inputs[i]:
                        float_inputs.append(1.0)
                    else:
                        float_inputs.append(0)
        return float_inputs

    new_X = []
    for person in range(len(X)):
        formatted_X = flatten_persons_inputs_for_model(X[person], means)
        new_X.append(formatted_X)
    new_X = np.array(new_X)

    # y:
    new_y = []
    for i in range(len(y)):
        if y[i] == ">50K" or y[i] == ">50K.":
            new_y.append(1)
        else:  # y[i] == "<=50k":
            new_y.append(0)

    new_y = np.array(new_y)

    return (new_X, new_y)


def generate_dataset(file_path):
    data = np.genfromtxt(file_path, delimiter=", ", dtype=str, autostrip=True)
    print("Data {} count: {}".format(file_path, len(data)))
    print(data[0])
    print(len(data[0]))

    means = find_means_for_continuous_types(data)
    print("Mean values for data types (if continuous): {}".format(means))

    X, y = prepare_data(data, means)
    print(X[0].shape)
    print(X[0])
    percent = sum([i for i in y]) * 1.0 / len(y)
    print("Data percentage {} that is >50k: {}%".format(file_path, percent * 100))

    return X.tolist(), y.tolist()


import torch


def main():
    train_output = "./data/synthetic/train/mytrain.json"
    test_output = "./data/synthetic/test/mytest.json"

    np.random.seed(0)
    A = np.random.binomial(n=1, p=0.5, size=10000)

    X1 = np.random.normal(loc=0, scale=1, size=10000)
    X2 = np.zeros(10000)
    for i in range(10000):
        if A[i] == 0:
            X2[i] = np.random.normal(loc=-1, scale=2)
        else:
            X2[i] = np.random.normal(loc=1, scale=2)

    Y = np.zeros(10000)
    for i in range(10000):
        if A[i] == 0:
            uh = 0.6
            ul = 0.3
            ha = ul * ((X1[i] + X2[i]) <= 0) + uh * ((X1[i] + X2[i]) > 0)
            # ha = 0.5 * ((X2[i]) <= 0) + 0.5 * ((X2[i]) > 0)
            Y[i] = np.random.binomial(n=1, p=ha)
        else:
            uh = 0.9
            ul = 0.1
            ha = ul * ((X1[i] + X2[i]) <= 0) + uh * ((X1[i] + X2[i]) > 0)
            Y[i] = np.random.binomial(n=1, p=ha)

    synthetic_dataset = np.column_stack((X1, X2, A, Y))
    # Split the Synthetic_dataset into train and test sets
    test_size = int(0.2 * synthetic_dataset.shape[0])
    indices = np.random.permutation(synthetic_dataset.shape[0])
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    test_set = synthetic_dataset[test_indices]
    train_set = synthetic_dataset[train_indices]

    # Split  based on the value of the A column
    train_set_0 = train_set[train_set[:, 0] < -0.5]
    train_set_1 = train_set[train_set[:, 0] >= -0.5]
    test_set_0 = test_set[test_set[:, 0] < -0.5]
    test_set_1 = test_set[test_set[:, 0] >= -0.5]

    X_train_set_0 = train_set_0[:, 0:-1].tolist()
    Y_train_set_0 = train_set_0[:, -1].tolist()

    X_train_set_1 = train_set_1[:, 0:-1].tolist()
    Y_train_set_1 = train_set_1[:, -1].tolist()

    X_test_set_0 = test_set_0[:, 0:-1].tolist()
    Y_test_set_0 = test_set_0[:, -1].tolist()

    X_test_set_1 = test_set_1[:, 0:-1].tolist()
    Y_test_set_1 = test_set_1[:, -1].tolist()

    # Create data structure
    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "num_samples": []}

    # for group 0 users
    train_len = len(X_train_set_0)
    print("training set for phd users: {}".format(train_len))
    test_len = len(X_test_set_0)
    uname = "group1"
    train_data["users"].append(uname)
    train_data["user_data"][uname] = {"x": X_train_set_0, "y": Y_train_set_0}
    train_data["num_samples"].append(train_len)
    test_data["users"].append(uname)
    test_data["user_data"][uname] = {"x": X_test_set_0, "y": Y_test_set_0}
    test_data["num_samples"].append(test_len)

    # for group 1 users

    train_len = len(X_train_set_1)
    print("training set for non-phd users: {}".format(train_len))
    test_len = len(X_test_set_1)
    uname = "group2"
    train_data["users"].append(uname)
    train_data["user_data"][uname] = {"x": X_train_set_1, "y": Y_train_set_1}
    train_data["num_samples"].append(train_len)
    test_data["users"].append(uname)
    test_data["user_data"][uname] = {"x": X_test_set_1, "y": Y_test_set_1}
    test_data["num_samples"].append(test_len)

    with open(train_output, "w") as outfile:
        json.dump(train_data, outfile)
    with open(test_output, "w") as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()
