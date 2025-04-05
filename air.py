import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Load Data
file_1= r"C:\Users\sar\OneDrive\Desktop\Airbnd data\Airbnd Data\listings.csv"
df=pd.read_csv(file_1)

# 📌 Data Cleaning
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)  # Convert price to float
df.dropna(subset=['price', 'bedrooms', 'room_type', 'number_of_reviews'], inplace=True)  # Drop missing values

# 📌 Feature Selection
X = df[['bedrooms', 'number_of_reviews']]  # Numeric Features
X = pd.get_dummies(df[['bedrooms', 'room_type']], drop_first=True)  # Encode categorical variables
y = df['price']  # Target Variable

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Predictions
y_pred = model.predict(X_test)

# 📌 Model Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 📌 Visualization (Actual vs Predicted)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Linear Regression)")
plt.show()
