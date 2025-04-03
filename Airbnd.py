import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Files ka path (Update karo agar zarurat ho)
file_listings = r"C:\Users\sar\OneDrive\Desktop\Airbnd data\Airbnd Data\listings.csv"
file_reviews = r"C:\Users\sar\OneDrive\Desktop\Airbnd data\Airbnd Data\reviews.csv"
file_calendar = r"C:\Users\sar\OneDrive\Desktop\Airbnd data\Airbnd Data\calendar.csv"

# DataFrames load karo
df_listings = pd.read_csv(file_listings)
df_reviews = pd.read_csv(file_reviews)
df_calendar = pd.read_csv(file_calendar)

# Pehli kuch rows dekho
print("📌 Listings Data:")
print(df_listings.head())

print("\n📌 Reviews Data:")
print(df_reviews.head())

print("\n📌 Calendar Data:")
print(df_calendar.head())

print("📌 Missing Values in Listings Data:\n :",df_listings.isnull().sum())

print("\n📌 Missing Values in Calendar Data:\n", df_calendar.isnull().sum())


# Price column ko clean aur numeric me convert karo
df_listings['price'] = df_listings['price'].replace('[\$,]', '', regex=True).astype(float)
df_listings['price'] = df_listings['price'].replace('[\$,]', '', regex=True)


# Fillna

df_listings.fillna({'reviews_per_month': 0, 'price': df_listings['price'].median()}, inplace=True)

# 🔹 Outliers remove  
df_listings = df_listings[df_listings['price'] < df_listings['price'].quantile(0.99)] 


# 🔹 Data standardize (Lowercase & strip spaces)
print(df_listings['neighbourhood'].head(10))
df_listings['neighbourhood'] = df_listings['neighbourhood'].str.strip().str.lower()


# Avg price per nighbourhood

avg_price_neighbourhood = df_listings.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x=avg_price_neighbourhood.index[:10],y=avg_price_neighbourhood.values[:10],palette='coolwarm')
plt.title('🔹 Top 10 Expensive Neighbourhood',fontsize=15)
plt.xticks(rotation =45)
plt.xlabel("Neighbourhood",fontsize=14)
plt.ylabel("Averge Price",fontsize=14)
plt.show()

top_hosts = df_listings.groupby('host_id')['review_scores_rating'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,5))
sns.barplot(x=top_hosts.index , y=top_hosts.values ,legend=False, palette='viridis')
plt.title("🔹 Top 10 Hosts with Highest Ratings",fontsize=14)
plt.xlabel("Host ID",fontsize=14)
plt.ylabel("Averge Ratings",fontsize=14)
plt.show()

#✅ 3. Correlation Analysis
plt.figure(figsize=(8,5)) 
sns.heatmap(df_listings[['price','availability_365','number_of_reviews']].corr(),annot=True,cmap='coolwarm')
plt.title("🔹 Correlation Heatmap",fontsize=14)
plt.show()


# #✅ 4. Price Distribution Across Property Types
plt.figure(figsize=(10,5))
sns.boxplot(x='room_type',y='price',data=df_listings, palette='coolwarm')
plt.title("🔹 Price Distribution by Room Type",fontsize=14)
plt.xlabel("Room Type",fontsize=14)
plt.ylabel("Price",fontsize=14)
plt.ylim(0,500) # Detect outliner
plt.show()

# 5. Price Trends Over Time (Seasonal Trends)
df_calendar['date'] = pd.to_datetime(df_calendar['date'])
df_calendar['price'] = df_calendar['price'].replace('[\$,]', '', regex=True).astype(float)

price_trend = df_calendar.groupby('date')['price'].mean()

plt.figure(figsize=(12, 5))
plt.plot(price_trend.index, price_trend.values, color='red', marker='o', linestyle='dashed')
plt.title('🔹 Price Trends Over Time',fontsize=14)
plt.xlabel('Date',fontsize=14)
plt.ylabel('Average Price',fontsize=14)
plt.grid()
plt.show()
