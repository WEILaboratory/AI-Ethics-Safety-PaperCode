import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
import random


def add_noise(Y, ratio=0.5):
    for i in range(len(Y)):
        rr = random.random()
        if rr < ratio:
            Y[i] = 1 - Y[i]
    return Y


def load_Bank(train_filename, test_filename):
    df_train = pd.read_csv(train_filename, encoding="latin-1")
    df_test = pd.read_csv(test_filename, encoding="latin-1")
    df = pd.concat([df_train, df_test], axis=0)

    for col_name in df.columns.tolist():
        df = df.drop(df[df[col_name] == "unknown"].index)

    df.drop(
        columns=[
            "month",
            "day_of_week",
            "duration",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ],
        inplace=True,
    )
    df = df.reset_index(drop=True)
    # OneHotEncoding of job column
    ohe = OneHotEncoder(sparse=False)
    # print(df["job"].unique())
    df2 = pd.DataFrame(
        ohe.fit_transform(df["job"].to_frame()),
        columns="job_" + np.sort(df["job"].unique()),
    )

    df = pd.concat(
        (df, df2),
        axis=1,
    )
    df.drop(columns=["job"], inplace=True)
    # print(df["marital"].unique())
    df.loc[(df.marital == "married"), "marital"] = int(1)
    df.loc[(df.marital == "single"), "marital"] = int(0)
    df.loc[(df.marital == "divorced"), "marital"] = int(0)
    # Marital column has 3 values lets apply OneHotEncoding again.

    # Default Column
    df.loc[(df.default == "yes"), "default"] = 1
    df.loc[(df.default == "no"), "default"] = 0

    # print(df["poutcome"].unique())

    df.loc[(df.housing == "yes"), "housing"] = 1  # housing column label encoding
    df.loc[(df.housing == "no"), "housing"] = 0

    # Loan column label encoding
    df.loc[(df.loan == "yes"), "loan"] = 1
    df.loc[(df.loan == "no"), "loan"] = 0

    # contact column label encoding
    df.loc[
        (df.contact == "telephone"), "contact"
    ] = 1  # 0 means cellular 1 means telephone
    df.loc[(df.contact == "cellular"), "contact"] = 0

    df.loc[(df.pdays == -1), "pdays"] = 999

    df.loc[(df.y == "yes"), "y"] = 1
    df.loc[(df.y == "no"), "y"] = 0

    df = pd.concat(
        (
            df,
            pd.DataFrame(
                ohe.fit_transform(df["poutcome"].to_frame()),
                columns="poutcome_" + np.sort(df["poutcome"].unique()),
            ),
        ),
        axis=1,
    )
    df.drop(columns=["poutcome"], inplace=True)

    df.loc[(df.education == "illiterate"), "education"] = 0
    df.loc[(df.education == "basic.4y"), "education"] = 1
    df.loc[(df.education == "basic.6y"), "education"] = 1
    df.loc[(df.education == "basic.9y"), "education"] = 1
    df.loc[(df.education == "high.school"), "education"] = 1
    df.loc[(df.education == "professional.course"), "education"] = 1
    df.loc[(df.education == "university.degree"), "education"] = 1
    # df = pd.concat(
    #     (
    #         df,
    #         pd.DataFrame(
    #             ohe.fit_transform(df["education"].to_frame()),
    #             columns="education_" + np.sort(df["education"].unique()),
    #         ),
    #     ),
    #     axis=1,
    # )
    # df.drop(columns=["education"], inplace=True)

    X = df.drop(["y"], axis=1)
    # print(X.info())
    Y = df[["y"]]
    return X, Y


def counterfacts_dataset(df):
    df_c = df.copy()
    unique_values = df_c["education"].unique()
    df_c["education"] = df_c["education"].map(
        {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]}
    )
    return df_c


def generate_bank_sample(D_init_num=1, D_providers_num=5, add_noise_bool=True):
    X, Y = load_Bank("dataset/Bank/train.csv", "dataset/Bank/test.csv")
    X_c = counterfacts_dataset(X)
    Y_c = Y.copy()

    X_ori = X.copy()
    Y_ori = Y.copy()
    X_ori_c = X_c.copy()
    Y_ori_c = Y_c.copy()

    X, X_test, Y, Y_test = train_test_split(X, Y_c, test_size=0.1, random_state=0)

    X_c, X_test_c, Y_c, Y_test_c = train_test_split(
        X_c, Y_c, test_size=0.1, random_state=0
    )

    D_init_x = []
    D_init_y = []
    D_providers_x = []
    D_providers_y = []
    D_test_x = [
        pd.concat(
            [X_test, X_test_c],
            axis=0,
        ).values.astype("float")
    ]
    D_test_y = [
        pd.concat(
            [Y_test, Y_test_c],
            axis=0,
        ).values.astype("int")
    ]

    for ii in range(D_init_num):
        bias_level = 0
        male_num = int(500 * bias_level)
        female_num = int(500 * (1 - bias_level))

        random_male_index = X.sample(male_num, replace=False).index
        random_female_index = X_c.sample(female_num, replace=False).index

        D_init_i = pd.concat(
            [
                X_ori.iloc[random_male_index].copy(),
                X_ori_c.iloc[random_female_index].copy(),
            ],
            axis=0,
        )
        D_init_i_y = pd.concat(
            [
                Y_ori.iloc[random_male_index].copy(),
                Y_ori_c.iloc[random_female_index].copy(),
            ],
            axis=0,
        )
        D_init_x.append(D_init_i.values.astype("float"))
        D_init_y.append(D_init_i_y.values.astype("int"))

    provider_bias = [0.01, 0.3, 0.5, 0.7, 0.99]
    # provider_bias = [0.0, 1.0]
    provider_num = [5000, 5000, 5000, 5000, 5000, 5000]
    for ii in range(D_providers_num):
        # fix bias
        bias_level = provider_bias[ii]
        # 随机bias
        # bias_level = random.choice(provider_bias)
        male_num = int(provider_num[ii] * bias_level)
        female_num = int(provider_num[ii] * (1 - bias_level))

        random_male_index = X.sample(male_num, replace=False).index
        random_female_index = X_c.sample(female_num, replace=False).index

        D_provide_i = pd.concat(
            [
                X_ori.iloc[random_male_index].copy(),
                X_ori_c.iloc[random_female_index].copy(),
            ]
        )
        D_provide_i_y = pd.concat(
            [
                Y_ori.iloc[random_male_index].copy(),
                Y_ori_c.iloc[random_female_index].copy(),
            ]
        )
        D_providers_x.append(D_provide_i.values.astype("float"))
        D_providers_y.append(D_provide_i_y.values.astype("int"))
    if add_noise_bool == True:
        D_init_y[0] = add_noise(D_init_y[0], 0.5)
        for ii in range(len(D_providers_y)):
            D_providers_y[ii] = add_noise(D_providers_y[ii], 0.2)

    # for ii in range(D_providers_num):
    #     predictor = MLPClassifier(hidden_layer_sizes=50, max_iter=1000)
    #     predictor.fit(D_providers_x[ii], D_providers_y[ii])

    #     y_pred = predictor.predict(D_test_x[0])

    #     acc = round(predictor.score(D_test_x[0], D_test_y[0]) * 100, 2)

    #     print(acc)

    #     y_pred_z1 = y_pred[np.where(D_test_x[0][:, 2] > 0)[0]]
    #     y_pred_z0 = y_pred[np.where(D_test_x[0][:, 2] <= 0)[0]]

    #     dp = np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(
    #         D_test_x[0][:, 2] > 0
    #     ) - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(D_test_x[0][:, 2] <= 0)

    #     print(dp)

    return D_init_x, D_init_y, D_providers_x, D_providers_y, D_test_x, D_test_y


if __name__ == "__main__":
    generate_bank_sample()
