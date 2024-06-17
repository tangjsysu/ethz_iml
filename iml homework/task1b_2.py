import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv("../data/train_1b.csv")
y = data["y"].to_numpy()
data = data.drop(columns=["Id", "y"])
# print a few data samples
print(data.head())

X = data.to_numpy()

Ridge_ = RidgeCV(alphas=np.arange(1, 1001, 100),
                 # scoring="neg_mean_squared_error",
                 store_cv_values=True,
                 # cv=5
                 ).fit(X, y)
# 无关交叉验证的岭回归结果
score = Ridge_.score(X, y)  # 0.6060251767338429

# 调用所有交叉验证的结果 --- 10个不同alpha下的交叉验证结果集(留一交叉验证)
cv_score = Ridge_.cv_values_
cv_score_mean = Ridge_.cv_values_.mean(axis=0)
# [0.52823795 0.52787439 0.52807763 0.52855759 0.52917958 0.52987689, 0.53061486 0.53137481 0.53214638 0.53292369]

# 查看最佳正则化系数
best_alpha = Ridge_.alpha_  # 101
print(best_alpha)