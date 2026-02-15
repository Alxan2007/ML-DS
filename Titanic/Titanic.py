from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
price_for_house = pd.read_csv('train.csv')
X = price_for_house.drop(columns=['SalePrice'])
y = price_for_house['SalePrice']

# Кодирование категориальных признаков
X = pd.get_dummies(X, drop_first=True)

# Проверка дубликатов
duplicates = X[X.duplicated()]
if len(duplicates) > 0:
    print("Найдены дубликаты:", len(duplicates))
else:
    print("Дубликатов нет")

# Заполнение пропусков
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)  # Сохраняем имена столбцов

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Масштабирование (для XGBoost и LinearRegression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Обучение моделей ---

# 1. XGBoost
model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
model_xgb.fit(X_train_scaled, y_train)
y_pred_xgb = model_xgb.predict(X_test_scaled)
r2_xgb = r2_score(y_test, y_pred_xgb)

# 2. Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=None)
model_rf.fit(X_train, y_train)  # Не требует масштабирования
y_pred_rf = model_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# 3. Linear Regression (с масштабированием)
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)

# --- Визуализация: Фактические vs Предсказанные ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Фактические vs Предсказанные значения', fontsize=16)

models = [
    ('XGBoost', y_pred_xgb, r2_xgb),
    ('Random Forest', y_pred_rf, r2_rf),
    ('Linear Regression', y_pred_lr, r2_lr)
]

for idx, (name, y_pred_model, r2_model) in enumerate(models):
    ax = axes[idx]
    ax.scatter(y_test, y_pred_model, alpha=0.6, color='teal', edgecolors='none', s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальные предсказания')
    ax.set_xlabel('Фактические значения')
    ax.set_ylabel('Предсказанные значения')
    ax.set_title(f'{name}\n$R^2 = {r2_model:.3f}$')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# Вывод метрик в консоль
print(f"XGBoost R²: {r2_xgb:.3f}")
print(f"Random Forest R²: {r2_rf:.3f}")
print(f"Linear Regression R²: {r2_lr:.3f}")
