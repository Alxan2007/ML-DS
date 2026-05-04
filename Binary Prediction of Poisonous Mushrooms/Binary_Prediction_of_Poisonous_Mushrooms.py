import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# Загрузка и первичный анализ
data = pd.read_csv("train 2.csv", nrows=100000)
print("Первые 5 строк:")
print(data.head())
print("\nИнформация о данных:")
print(data.info())
print("\nКоличество пропусков в столбцах:")
print(data.isna().sum())

# Проверка дубликатов
duplicates = data[data.duplicated()]
print("\nДубликаты:" if len(duplicates) > 0 else "\nНет дубликатов")
if len(duplicates) > 0:
    print(duplicates)

# Удаление нерелевантных столбцов
cols_to_drop = ['id', 'ring-type', 'cap-surface', 'gill-attachment',
               'gill-spacing', 'stem-root', 'stem-surface',
               'spore-print-color', 'veil-type', 'veil-color']
data.drop(cols_to_drop, axis=1, inplace=True)

# Обработка пропусков
# Для does-bruise-or-bleed заполняем пропуски наиболее частым значением
if 'does-bruise-or-bleed' in data.columns:
    most_frequent_bruise = data['does-bruise-or-bleed'].mode()[0] if not data['does-bruise-or-bleed'].mode().empty else 't'
    data['does-bruise-or-bleed'] = data['does-bruise-or-bleed'].fillna(most_frequent_bruise)

# Для has-ring аналогично
if 'has-ring' in data.columns:
    most_frequent_ring = data['has-ring'].mode()[0] if not data['has-ring'].mode().empty else 't'
    data['has-ring'] = data['has-ring'].fillna(most_frequent_ring)

# Кодирование категориальных переменных
data['class'] = data['class'].map({'e': 0, 'p': 1}).astype(int)

# Кодируем и заполняем пропуски для does-bruise-or-bleed
data['does-bruise-or-bleed'] = data['does-bruise-or-bleed'].map({'t': 1, 'f': 0}).fillna(0).astype(int)
data['has-ring'] = data['has-ring'].map({'t': 1, 'f': 0}).fillna(0).astype(int)

# One-Hot Encoding
categorical_cols = ['cap-shape', 'cap-color', 'gill-color', 'stem-color', 'habitat', 'season']
data = pd.get_dummies(data, columns=categorical_cols, prefix=categorical_cols)

print(data.isna().sum())
print(data.info())
print(data.head())

imputer = SimpleImputer(strategy='most_frequent')
data_imputer = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
data = data_imputer


X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_test_sclaer = scaler.transform(X_test)
X_train_scaler = scaler.transform(X_train)

model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train_scaler, y_train)
y_pred_lg = model_lr.predict(X_test_sclaer)
y_prob_lr = model_lr.predict_proba(X_test_sclaer)[:, 1]  # вероятности для класса 1

model_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model_rf.fit(X_train_scaler, y_train)
y_pred_rf = model_rf.predict(X_test_sclaer)
y_prob_rf = model_rf.predict_proba(X_test_sclaer)[:, 1]

print(y_pred_lg)
print(y_pred_rf)
print()
print("Логистическая регрессия:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_lg)*100}%')
print(f'Classification_report: {classification_report(y_test, y_pred_lg)}')

print('Random Forest:')
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)*100}%')
print(f'Classification_report: {classification_report(y_test, y_pred_rf)}')

# 12. Строим ROC-AUC кривые
plt.figure(figsize=(12, 5))

# ROC для логистической регрессии
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
plt.subplot(1, 2, 1)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Logistic Regression')
plt.legend()

# ROC для случайного леса
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
plt.subplot(1, 2, 2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest')
plt.legend()

plt.tight_layout()
plt.show()

# 13. Строим confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix для логистической регрессии
cm_lr = confusion_matrix(y_test, y_pred_lg)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Съедобный', 'Ядовитый'])
disp_lr.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix — Logistic Regression')

# Confusion matrix для случайного леса
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Съедобный', 'Ядовитый'])
disp_rf.plot(ax=axes[1], cmap='Greens')
axes[1].set_title('Confusion Matrix — Random Forest')

plt.tight_layout()
plt.show()


