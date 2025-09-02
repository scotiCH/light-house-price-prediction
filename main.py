import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

df = None
try:
    df = pd.read_csv('house_data.csv')
except FileNotFoundError:
    print("Ошибка: файл не найден.")
    print("Пожалуйста, создайте его и скопируйте в него предоставленные данные.")


print("Первые 5 строк набора данных: ")
print(df.head())
print("\nОписание данных: ")
print(df.describe())

X = df[['size_sqm', 'bedrooms', 'renovation', 'age']]
y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

print("\n Размеры выборки: ")
print(f"Размер обучающей выборки (X_train): {X_train.shape}")
print(f"Размер тестовой выборки (X_test): {X_test.shape}")

model = LinearRegression()

model.fit(X_train, y_train)

print("\nМодель обучена.")
print(f"Коэффициент модели (веса признаков): {model.coef_}")
print(f"Свободный член (intercept): {model.intercept_}")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nОценка производительности модели: ")
print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
print(f"Корень из среднеквадратичной ошибки(RMSE): {rmse:.2f}")
print(f"Коэффициент детерминации (R2-score): {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha = 0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=3)
plt.title("Сравнение реальных и предсказанных цен ")
plt.xlabel("Реальные цены")
plt.ylabel("Предсказанные цены")
plt.grid(True)
plt.show()

new_house_data = pd.DataFrame([[120, 2, 1, 1]], columns=X.columns)
predicted_price = model.predict(new_house_data)
print(f"\nПредсказанная цена для нового дома: {predicted_price.item():,.2f} рублей")