import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import copy
import matplotlib.pyplot as plt
from scipy.stats import t

os.chdir("/data/jiashigao/code/bias_query/")
os.getcwd()

df = pd.read_csv("dataset/COMPAS/compas-scores-two-years_processed.csv")
# 提取特征和目标变量
X = df[["age", "c_charge_degree", "race", "sex", "priors_count"]]
y = df[["two_year_recid"]]

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# X_test: sex=0
X_test_0 = X_test.assign(sex=0)

# X_test: sex=1
X_test_1 = X_test.assign(sex=1)

# 训练决策树模型
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

record_number = X_test.shape[0]


initial_size = int(0.05 * record_number)
e_sample_size = 20
repeat_times = 100

j_max = int((0.3 * record_number - initial_size) // e_sample_size)

true_p_z0_y1 = np.count_nonzero(clf.predict(X_test_0) == 1) / X_test_0.shape[0]
true_p_z1_y1 = np.count_nonzero(clf.predict(X_test_1) == 1) / X_test_1.shape[0]

true_dp = np.abs(true_p_z0_y1 - true_p_z1_y1)


random_index = np.random.permutation(X_test_0.index)
X_test_0_rand = X_test_0.reindex(random_index)
X_test_1_rand = X_test_1.reindex(random_index)

old_size = 0
current_size = initial_size
epsilon = 0.01
significance_level = 1e-3
while abs(old_size - current_size) != 0:
    # initial samples
    X_test_0_samples = X_test_0_rand[:current_size]
    X_test_1_samples = X_test_1_rand[:current_size]

    p_z0_y1 = (
        np.count_nonzero(clf.predict(X_test_0_samples) == 1) / X_test_0_samples.shape[0]
    )
    p_z1_y1 = (
        np.count_nonzero(clf.predict(X_test_1_samples) == 1) / X_test_1_samples.shape[0]
    )

    dp = np.abs(p_z0_y1 - p_z1_y1)

    dp_error = np.abs(dp - true_dp)

    t_value = t.ppf(significance_level / 2, current_size - 1)

    A0 = -t_value

    S0 = np.sqrt(p_z1_y1 * (1 - p_z1_y1))
    old_size = current_size
    current_size = int(A0 * S0 / epsilon)
    print(current_size)
print("end")
