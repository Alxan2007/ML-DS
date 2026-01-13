import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV‑файла в DataFrame
df = pd.read_csv('sales.csv')
df['revenue'] = df['price'] * df['quantity']
most_expensive = df.loc[df['price'].idxmax()]
# Проверка на наличие дубликатов в данных
duplicates = df[df.duplicated()]
if len(duplicates) > 0:
    print('Duplicates')
else:
    print('No duplicates')
print(df.isnull().sum())
print("Данные о продажах:")
print(df)
# Вывод таблицы с названиями товаров и соответствующей выручкой
print("\nОбщая выручка по товарам:")
print(df[['product', 'revenue']])
# Вывод информации о самом дорогом товаре
print("\nСамый дорогой товар:")
print(f"Название: {most_expensive['product']}")
print(f"Цена за единицу: {most_expensive['price']}")

plt.figure(figsize=(10, 6))
plt.bar(df['product'], df['revenue'], color='skyblue', edgecolor='navy', alpha=0.7)
plt.title('Выручка по товарам', fontsize=16, fontweight='bold')
plt.xlabel('Товар', fontsize=12)
plt.ylabel('Выручка (руб.)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
