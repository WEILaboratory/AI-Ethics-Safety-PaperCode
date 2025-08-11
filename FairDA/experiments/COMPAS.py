import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import random
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def add_noise(Y, ratio=0.5):
    for i in range(len(Y)):
        rr = random.random()
        if rr < ratio:
            Y[i] = 1 - Y[i]
    return Y


def load_Compas(filename):
    df = pd.read_csv(filename, encoding="latin-1")
    # delete null rows
    df = df.dropna()

    unique_values = df["sex"].unique()
    df["sex"] = df["sex"].map(
        {unique_values[0]: unique_values[1], unique_values[1]: unique_values[0]}
    )

    X = df.drop(["is_recid", "Unnamed: 0"], axis=1)

    Y = df[["is_recid"]]
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


def generate_Compas_sample(D_init_num=1, D_providers_num=5, add_noise_bool=True):
    X_ori, Y_ori = load_Compas("dataset/COMPAS/compas-scores-two-years_processed.csv")
    X_ori_c = counterfacts_dataset(X_ori)
    Y_ori_c = Y_ori.copy()

    X, X_test, Y, Y_test = train_test_split(X_ori, Y_ori, test_size=250, random_state=0)

    X_c, X_test_c, Y_c, Y_test_c = train_test_split(
        X_ori_c, Y_ori_c, test_size=250, random_state=0
    )
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
        bias_level = 0
        black_num = int(500 * bias_level)
        white_num = int(500 * (1 - bias_level))

        random_black_index = X.sample(black_num, replace=False).index
        random_white_index = X_c.sample(white_num, replace=False).index

        D_init_i = pd.concat(
            [
                X_ori.iloc[random_black_index].copy(),
                X_ori_c.iloc[random_white_index].copy(),
            ],
            axis=0,
        )
        D_init_i_y = pd.concat(
            [
                Y_ori.iloc[random_black_index].copy(),
                Y_ori_c.iloc[random_white_index].copy(),
            ],
            axis=0,
        )
        D_init_x.append(D_init_i.values)
        D_init_y.append(D_init_i_y.values)

    # provider_bias = [0.0, 0.1, 0.2, 0.9, 1.0]
    provider_bias = [0.1, 0.3, 0.5, 0.7, 0.9]
    provider_num = [2000, 2000, 2000, 2000, 2000]
    for ii in range(D_providers_num):
        # fix bias
        bias_level = provider_bias[ii]
        # 随机bias
        # bias_level = random.choice(provider_bias)
        black_num = int(provider_num[ii] * bias_level)
        white_num = int(provider_num[ii] * (1 - bias_level))

        random_black_index = X.sample(black_num, replace=False).index
        random_white_index = X_c.sample(white_num, replace=False).index

        D_provide_i = pd.concat(
            [
                X_ori.iloc[random_black_index].copy(),
                X_ori_c.iloc[random_white_index].copy(),
            ]
        )
        D_provide_i_y = pd.concat(
            [
                Y_ori.iloc[random_black_index].copy(),
                Y_ori_c.iloc[random_white_index].copy(),
            ]
        )
        D_providers_x.append(D_provide_i.values)
        D_providers_y.append(D_provide_i_y.values)
    if add_noise_bool == True:
        D_init_y[0] = add_noise(D_init_y[0], 0.5)
        for ii in range(len(D_providers_y)):
            D_providers_y[ii] = add_noise(D_providers_y[ii], 0.4)
            
    # for ii in range(D_providers_num):
    #     predictor = XGBClassifier()
    #     predictor.fit(D_providers_x[ii], D_providers_y[ii])
    #     y_pred = predictor.predict(D_test_x[0])

    #     acc = round(predictor.score(D_test_x[0], D_test_y[0]) * 100, 2)

    #     print(acc)

    #     y_pred_z1 = y_pred[np.where(D_test_x[0][:, 4] > 0)[0]]
    #     y_pred_z0 = y_pred[np.where(D_test_x[0][:, 4] < 0)[0]]

    #     dp = np.count_nonzero(y_pred_z1 == 1) / np.count_nonzero(
    #         D_test_x[0][:, 4] > 0
    #     ) - np.count_nonzero(y_pred_z0 == 1) / np.count_nonzero(D_test_x[0][:, 4] < 0)

    #     print(dp)
    return D_init_x, D_init_y, D_providers_x, D_providers_y, D_test_x, D_test_y

    # testing samples 500
    # random_black_index = X.sample(n=int(1 * 1000), replace=False).index
    # random_white_index = X_c.sample(n=int(1 * 1000), replace=False).index

    # unique_values = X["race"].unique()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.3, random_state=0
    # )
    # model = XGBClassifier()
    # model_importance = model.fit(X_train.values, y_train.values)
    # pred = model.predict(X_test.values)
    # acc = accuracy_score(y_test, pred)
    # print("function end")


if __name__ == "__main__":
    (
        D_init_x,
        D_init_y,
        D_providers_x,
        D_providers_y,
        D_test_x,
        D_test_y,
    ) = generate_Compas_sample(D_init_num=1, D_providers_num=2)

    print("end")
