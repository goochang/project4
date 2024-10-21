import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
titanic = sns.load_dataset('titanic')
# print(titanic.head())
# print(titanic.describe())
# count : 결측치 값을 제외한 데이터의 개수를 나타냅니다.
# mean : 데이터의 평균값을 나타냅니다.
# std : 데이터의 표준편차를 나타냅니다.
# min : 해당 열에서 가장 작은 값을 나타냅니다.
# max : 해당 열에서 가장 큰 값을 나타냅니다.
# 25% : 데이터의 첫 번째 사분위수를 나타냅니다.
# 50% : 데이터의 중앙 값을 나타냅니다.
# 75% : 데이터의 세 번째 사분위수을 나타냅니다.

# print(titanic["embarked"].mode().iloc[0])
# 결측치 age 177 / deck 688 / embark_town 2

titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["embarked"] = titanic["embarked"].fillna(titanic["embarked"].mode().iloc[0])
# print(titanic.isnull().sum())

titanic["sex"] = titanic["sex"].map({'male':0, 'female':1})
titanic["alive"] = titanic["alive"].map({'no':0, 'yes':1})
titanic["embarked"] = titanic["embarked"].map({'C':0, 'Q':1, 'S':2})

# print(titanic["sex"].head())
# print(titanic["alive"].head())
# print(titanic["embarked"].head())

titanic["family_size"] = titanic["sibsp"] + titanic["parch"] + 1 # 본인포함
# print(titanic["survived"])

x = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size']]  # feature
y = titanic['survived']  # target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# --- #

# Random Forest 모델 생성 및 학습
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# --- #
# XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 모델 학습
xgb_model.fit(X_train, y_train)

# 예측
y_pred_xgb = xgb_model.predict(X_test)

# 평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')