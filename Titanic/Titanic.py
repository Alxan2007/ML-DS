import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier


titanic = pd.read_csv('Titanic/titanic.csv')



titanic.drop('Cabin', axis=1, inplace=True)


duplicates = titanic[titanic.duplicated()]
if len(duplicates) > 0:
    print(duplicates)
else:
    print('No Duplicates')


titanic['family_size'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic = titanic.dropna(subset=['Age'])

X = titanic[['Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'Embarked']]
y = titanic['Survived']

X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
logreg = RandomForestClassifier(n_estimators=100, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\nМетрики логистической регрессии:")
print(accuracy)
print(precision)
print(recall)
print(f1)

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


from matplotlib import pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC-кривая {roc_auc:.2f}', lw = 2)
plt.plot([0, 1], [0 , 1], color = 'navy', lw = 2, linestyle = '--', label = "Случайные числа")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()




























