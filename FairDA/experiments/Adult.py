import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import random


def add_noise(Y, ratio=0.5):
    for i in range(len(Y)):
        rr = random.random()
        if rr < ratio:
            Y[i] = 1 - Y[i]
    return Y


def load_Adult(filename):
    df = pd.read_csv(filename, encoding="latin-1")
    # Mapping binary values to the expected output
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    # Replacing question marks in dataset with null values
    df.replace("?", np.nan)
    # Since mode is Prof-specialty, replacing null values with it
    df["occupation"] = df["occupation"].fillna("Prof-specialty")
    # Since mode is Private, replacing null values with it
    df["workclass"] = df["workclass"].fillna("Private")
    # Since mode is United-States, replacing null values with it
    df["native.country"] = df["native.country"].fillna("United-States")
    # Since it has 0 correlation, it can be dropped
    df.drop(["fnlwgt"], axis=1, inplace=True)

    dataset = df.copy()
    # Distributing Age column in 3 significant parts and plotting it corresponding to the output feature(income)

    dataset["age"] = pd.cut(
        dataset["age"], bins=[0, 25, 50, 100], labels=["Young", "Adult", "Old"]
    )
    # Capital gain and capital loss can be combined and transformed into a feature capital difference. Plotting the new feature corresponding to income

    dataset["Capital Diff"] = dataset["capital.gain"] - dataset["capital.loss"]
    dataset.drop(["capital.gain"], axis=1, inplace=True)
    dataset.drop(["capital.loss"], axis=1, inplace=True)

    dataset["Capital Diff"] = pd.cut(
        dataset["Capital Diff"], bins=[-5000, 5000, 100000], labels=["Minor", "Major"]
    )
    dataset["Hours per Week"] = pd.cut(
        dataset["hours.per.week"],
        bins=[0, 30, 40, 100],
        labels=["Lesser Hours", "Normal Hours", "Extra Hours"],
    )
    df.drop(["education.num"], axis=1, inplace=True)
    df["education"].replace(
        ["11th", "9th", "7th-8th", "5th-6th", "10th", "1st-4th", "Preschool", "12th"],
        " School",
        inplace=True,
    )
    df["race"].unique()
    df["race"].replace(
        ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        " Other",
        inplace=True,
    )

    # Combining all other into one class

    countries = np.array(dataset["native.country"].unique())
    countries = np.delete(countries, 0)
    dataset["native.country"].replace(countries, "Other", inplace=True)
    df["native.country"].replace(countries, "Other", inplace=True)

    # Splitting the data set into features and outcome

    X = df.drop(["income"], axis=1)
    Y = df[["income"]]

    categorical = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]
    for feature in categorical:
        le = preprocessing.LabelEncoder()
        X[feature] = le.fit_transform(X[feature])

    scaler = StandardScaler()

    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, Y


def counterfacts_dataset(df):
    df_c = df.copy()
    unique_values = df_c["sex"].unique()
    df_c["sex"] = df_c["sex"].map(
        {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]}
    )
    return df_c


def generate_Adult_sample(
    D_init_num=1, D_providers_num=5, add_noise_bool=True, coverage=0.001):
    X, Y = load_Adult("dataset/Adult/adult.csv")
    X_c = counterfacts_dataset(X)
    Y_c = Y.copy()

    X_ori = X.copy()
    Y_ori = Y.copy()
    X_ori_c = X_c.copy()
    Y_ori_c = Y_c.copy()

    X, X_test, Y, Y_test = train_test_split(X, Y_c, test_size=0.1)

    X_c, X_test_c, Y_c, Y_test_c = train_test_split(X_c, Y_c, test_size=0.1)

    D_init_x = []
    D_init_y = []
    D_providers_x = []
    D_providers_y = []
    D_test_x = [
        pd.concat(
            [X_test, X_test_c],
            axis=0,
        ).values
    ]
    D_test_y = [
        pd.concat(
            [Y_test, Y_test_c],
            axis=0,
        ).values
    ]

    for ii in range(D_init_num):
        bias_level = 0.0
        male_num = int(1000 * bias_level)
        female_num = int(1000 * (1 - bias_level))

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
        D_init_x.append(D_init_i.values)
        D_init_y.append(D_init_i_y.values)

    provider_bias = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    # adding coverage
    X, X_test, Y, Y_test = train_test_split(X, Y_c, test_size=coverage)

    X_c, X_test_c, Y_c, Y_test_c = train_test_split(X_c, Y_c, test_size=coverage)
    # provider_bias = [0.0, 1.0]
    provider_num = [10000, 10000, 10000, 10000, 10000, 10000]

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
        D_providers_x.append(D_provide_i.values)
        D_providers_y.append(D_provide_i_y.values)

    if add_noise_bool == True:
        D_init_y[0] = add_noise(D_init_y[0], 0.5)
        for ii in range(len(D_providers_y)):
            D_providers_y[ii] = add_noise(D_providers_y[ii], 0.3)

    # for ii in range(D_providers_num):
    #     predictor = MLPClassifier(max_iter=100)
    #     predictor.fit(D_providers_x[ii], D_providers_y[ii])

    #     y_pred = predictor.predict(D_test_x[0])

    #     acc = round(predictor.score(D_test_x[0], D_test_y[0]) * 100, 2)

    #     print(acc)

    #     y_pred_z1 = y_pred[np.where(D_test_x[0][:, -5] > 0)[0]]
    #     y_pred_z0 = y_pred[np.where(D_test_x[0][:, -5] < 0)[0]]

    #     dp = np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(
    #         D_test_x[0][:, -5] > 0
    #     ) - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(D_test_x[0][:, -5] < 0)

    #     print(dp)

    return D_init_x, D_init_y, D_providers_x, D_providers_y, D_test_x, D_test_y


if __name__ == "__main__":
    generate_Adult_sample()
