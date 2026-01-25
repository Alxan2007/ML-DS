import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler


fetch_house = fetch_california_housing()
X = pd.DataFrame(fetch_house.data, columns=fetch_house.feature_names)
y = fetch_house.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.isnull().any().any())
print("Есть inf в X_train?", np.isinf(X_train).any().any())
print("Статистика X_train:")
print(X_train.describe())

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_random_forest_model = r2_score(y_test, y_pred)
print(r2_random_forest_model)
#Random Forest также показал сильный результат (R^2 = 0.805), что обусловлено его способностью эффективно работать
# с нелинейными зависимостями и взаимодействиями признаков без необходимости предварительного масштабирования.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_ridg = Ridge(alpha=1.0)
model_ridg.fit(X_train_scaled, y_train)
y_pred_ridg = model_ridg.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred_ridg)
print(r2)
# Ridge-регрессия достигла более скромного результата (R^2 = 0.576), что объясняется линейной природой модели,
# неспособной адекватно описывать сложные зависимости в данных, несмотря на применение стандартизации признаков и регуляризации.

model_xgboost = xgb.XGBRegressor(learning_rate=0.1, n_estimators=500, max_depth=6, random_state=42, colsample_bytree=0.8, subsample=0.8)
model_xgboost.fit(X_train, y_train)
y_pred_xgboost = model_xgboost.predict(X_test)
accuracy_3 = model_xgboost.score(X_test, y_test)
print(accuracy_3)
# Наилучшую предсказательную способность продемонстрировал XGBoost с коэффициентом детерминации R^2
# = 0.8489, что указывает на высокую точность модели в объяснении дисперсии целевой переменной — медианной стоимости домов.


# Вывод: Таким образом, для данной задачи наиболее подходящей моделью является XGBoost, сочетающий высокую точность,
# устойчивость к переобучению и гибкость в учёте нелинейных паттернов.


