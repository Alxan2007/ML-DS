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








