import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



titanic = pd.read_csv('titanic.csv')


titanic['Age'] = titanic['Age'].fillna(
    titanic.groupby(['Sex', 'Pclass'])['Age'].transform('median')
)
titanic.drop('Cabin', axis=1, inplace=True)


duplicates = titanic[titanic.duplicated()]
if len(duplicates) > 0:
    print(duplicates)
else:
    print('No Duplicates')


titanic['family_size'] = titanic['SibSp'] + titanic['Parch'] + 1

print(titanic.isnull().sum())
print((titanic.isnull().mean() * 100).round(1))




X = titanic[['Pclass', 'Sex', 'Age', 'Fare', 'family_size', 'Embarked']]
y = titanic['Survived']

X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
logreg = LogisticRegression(max_iter=300, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nМетрики логистической регрессии:")
print(accuracy)
print(precision)
print(recall)
print(f1)


from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, y_pred).plot()
plt.title("Матрица ошибок")
plt.savefig('survival_plot.pdf', bbox_inches='tight')
plt.show()



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-score': f1_score(y_test, y_pred)
}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values())
plt.ylabel("Score")
plt.title("Classification Metrics")
plt.ylim(0, 1)
plt.savefig('survival_plot.pdf', bbox_inches='tight')
plt.show()





















