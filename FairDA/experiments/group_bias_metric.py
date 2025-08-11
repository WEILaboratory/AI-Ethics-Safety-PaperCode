import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import copy

os.chdir("/data/jiashigao/code/bias_query/")
os.getcwd()

df = pd.read_csv("dataset/COMPAS/compas-scores-two-years_processed.csv")
print(df.info())

# 提取特征和目标变量
X = df[["age", "c_charge_degree", "race", "sex", "priors_count"]]
y = df[["two_year_recid"]]

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


i_bias_sex_list = []
i_bias_race_list = []

g_bias_sex_list = []
g_bias_race_list = []


# 训练决策树模型
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# train_accuracy = clf.score(X_train, y_train)

# print("训练集上的准确率: {:.2f}".format(train_accuracy))

test_accuracy = clf.score(X_test, y_test)
y_predict = clf.predict(X_test)
print("测试训练集上的准确率: {:.2f}".format(test_accuracy))

# sex属性进行反转处理
X_test_sex_reverse = copy.deepcopy(X_test)
X_test_sex_reverse["sex"] = X_test_sex_reverse["sex"].apply(lambda x: 1 - x)
# race属性进行反转处理
X_test_race_reverse = copy.deepcopy(X_test)
X_test_race_reverse["race"] = X_test_race_reverse["race"].apply(lambda x: 1 - x)

# Y=1
X_test_y1 = X_test.loc[y_test.loc[y_test["two_year_recid"] == 1].index]
X_test_y0 = X_test.loc[y_test.loc[y_test["two_year_recid"] == 0].index]
y_test1 = y_test.loc[y_test["two_year_recid"] == 1]
y_test0 = y_test.loc[y_test["two_year_recid"] == 0]

# group bias 计算
# positive Equal Opportunity
X_test_y1_a1 = copy.deepcopy(X_test_y1)
X_test_y1_a1["sex"] = 1

X_test_y1_a0 = copy.deepcopy(X_test_y1)
X_test_y1_a0["sex"] = 0

g_bias_PEO = (
    np.count_nonzero(clf.predict(X_test_y1_a1) == 1) / X_test_y1.shape[0]
    - np.count_nonzero(clf.predict(X_test_y1_a0) == 1) / X_test_y1.shape[0]
)
# Negative Equal Opportunity
X_test_y0_a1 = copy.deepcopy(X_test_y0)
X_test_y0_a1["sex"] = 1

X_test_y0_a0 = copy.deepcopy(X_test_y0)
X_test_y0_a0["sex"] = 0

g_bias_NEO = (
    np.count_nonzero(clf.predict(X_test_y0_a1) == 0) / X_test_y0.shape[0]
    - np.count_nonzero(clf.predict(X_test_y0_a0) == 0) / X_test_y0.shape[0]
)
# Positive Mis-Equal Opportunity
X_test_y0_a1 = copy.deepcopy(X_test_y0)
X_test_y0_a1["sex"] = 1

X_test_y0_a0 = copy.deepcopy(X_test_y0)
X_test_y0_a0["sex"] = 0

g_bias_PMEO = (
    np.count_nonzero(clf.predict(X_test_y1_a1) == 1) / X_test_y0.shape[0]
    - np.count_nonzero(clf.predict(X_test_y1_a0) == 1) / X_test_y0.shape[0]
)

#
X_test_sex_0 = X_test.loc[X_test["sex"] == 0]
X_test_sex_1 = X_test.loc[X_test["sex"] == 1]

g_bias_sex = np.mean(np.abs(clf.predict(X_test_sex_0))) / np.mean(
    np.abs(clf.predict(X_test_sex_1))
)

X_test_race_0 = X_test.loc[X_test["race"] == 0]
X_test_race_1 = X_test.loc[X_test["race"] == 1]

g_bias_race = np.mean(np.abs(clf.predict(X_test_race_0))) / np.mean(
    np.abs(clf.predict(X_test_race_1))
)

g_bias_sex_list.append(g_bias_sex)
g_bias_race_list.append(g_bias_race)

print("g_bias_sex: {}, g_bias_race".format(g_bias_sex, g_bias_race))


fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(i_bias_sex_list)
ax.plot(i_bias_race_list)
ax.set_xlabel("append amount")
ax.set_ylabel("indiv-bias")
plt.savefig("fig/result.jpg")
print("end")
