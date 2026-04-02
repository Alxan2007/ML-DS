import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('bike-sharing-demand/train.csv')
print(data.head())


print(data.isnull().sum())
print(data.dtypes)

duplicates = data[data.duplicated()]
if len(duplicates) > 0:
    print(duplicates.head())
else:
    print('No duplicate rows\n')

print(data.dtypes)

def extract_datetime_features(df, drop_original = True):

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    if drop_original:
        df.drop(['datetime'], axis=1, inplace=True)
    return df

data = extract_datetime_features(data, drop_original = True)

print("\nНовые столбцы после извлечения признаков:")
print(data[['year', 'month', 'day', 'hour', 'weekday']].head())

print("\nТипы данных после преобразования:")
print(data.dtypes)
print()

correlation_matrix = data.corr()
print(correlation_matrix['count'].sort_values(ascending = False))

print(correlation_matrix)
features_to_drop = ['count', 'month', 'is_weekend', 'casual', 'registered']
X = data.drop(features_to_drop, axis = 1)
y = data['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

model_rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)



import numpy as np
from sklearn.metrics import mean_squared_error
RMSE_lr = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE_LR: {RMSE_lr}")

RMSE_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"RMSE_RF: {RMSE_rf}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.scatter(y_test, y_pred, alpha=0.5, label='Linear Regression', color='blue')
plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest', color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение предсказаний моделей с реальными данными')
plt.legend()
plt.show()




