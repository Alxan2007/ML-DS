from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

price_for_house = pd.read_csv('train.csv')
X = price_for_house.drop(columns= ['SalePrice'])
y = price_for_house['SalePrice']

X = pd.get_dummies(X, drop_first=True)

duplicates = price_for_house[price_for_house.duplicated()]
if len(duplicates) > 0:
    print(duplicates)
else:
    print('No duplicates')


imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(r2)

regressor = RandomForestRegressor(n_estimators=100, random_state=42,max_depth=None)
regressor.fit(X_train, y_train)
y_pred_regressor = regressor.predict(X_test)
r2_regressor = r2_score(y_test, y_pred_regressor)
print(r2_regressor)

regressor_1 = LinearRegression()
regressor_1.fit(X_train, y_train)
y_pred_regressor_1 = regressor_1.predict(X_test)
r2_regressor_1 = r2_score(y_test, y_pred_regressor_1)
print(r2_regressor_1)


from matplotlib import pyplot as plt

y_pred_xgb = model.predict(X_test_scaled)          # XGBoost
y_pred_rf = regressor.predict(X_test)              # Random Forest
y_pred_lr = regressor_1.predict(X_test)            # Linear Regression

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Сравнение моделей: истинные vs предсказанные цены', fontsize=16)

models = ['XGBoost', 'RandomForestRegressor', 'LinearRegression']
y_preds = [y_pred_xgb, y_pred_rf, y_pred_lr]
r2_scores = [r2, r2_regressor, r2_regressor_1]
colors = ['#FF9999', '#66B2FF', '#99FF99']

for i, ax in enumerate(axes):
    ax.scatter(y_test, y_preds[i], alpha=0.6, color=colors[i], edgecolors='k', s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальное предсказание')
    ax.set_xlabel('Истинная цена дома ($)')
    ax.set_ylabel('Предсказанная цена ($)')
    ax.set_title(f'{models[i]}\nR² = {r2_scores[i]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()





