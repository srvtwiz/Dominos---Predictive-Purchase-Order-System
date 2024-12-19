import pandas as pd
import numpy as np
from warnings import filterwarnings
import matplotlib.pyplot as plt
import seaborn as sns
filterwarnings('ignore')

# Load the data
ing=pd.read_csv('data/Pizza_ingredients - Pizza_ingredients.csv')
sale=pd.read_csv('data/Pizza_Sale - pizza_sales.csv')

print(sale.info())
print(ing.info())

def convert_dates(date):
  for fmt in ('%d-%m-%Y', '%d/%m/%Y'):
    try:
      return pd.to_datetime(date, format=fmt)
    except ValueError:
      pass
  raise ValueError(f'no valid date format found for {date}')

sale['order_date'] = sale['order_date'].apply(convert_dates)

#handling missing values

sale['total_price']=sale['total_price'].fillna(sale['quantity']*sale['unit_price'])

#fill missing pizza_ingredients values by comparing pizza_name
ing_mapping = sale[['pizza_name', 'pizza_ingredients']].dropna().drop_duplicates()
ing_mapping = ing_mapping.set_index('pizza_name')['pizza_ingredients'].to_dict()
sale['pizza_ingredients'] = sale['pizza_ingredients'].fillna(sale['pizza_name'].map(ing_mapping))

# Fill missing `pizza_name_id` in sale by comparing `pizza_ingredients`
name_id_mapping = sale[['pizza_name', 'pizza_name_id']].dropna().drop_duplicates()
name_id_mapping = name_id_mapping.set_index('pizza_name')['pizza_name_id'].to_dict()
sale['pizza_name_id'] = sale['pizza_name_id'].fillna(sale['pizza_name'].map(name_id_mapping))

# Fill missing `pizza_name` in sales_df by `pizza_name_id`
ing_name_mapping = sale[['pizza_ingredients', 'pizza_name']].dropna().drop_duplicates()
ing_name_mapping = ing_name_mapping.set_index('pizza_ingredients')['pizza_name'].to_dict()
sale['pizza_name'] = sale['pizza_name'].fillna(sale['pizza_ingredients'].map(ing_name_mapping))

#fill missing pizza_catogeroy with pizza_name_id
category_mapping = sale[['pizza_name_id', 'pizza_category']].dropna().drop_duplicates()
category_mapping = category_mapping.set_index('pizza_name_id')['pizza_category'].to_dict()
sale['pizza_category'] = sale['pizza_category'].fillna(sale['pizza_name_id'].map(category_mapping))
print(sale.info())

# Fill missing `Items_qty_In_Grams` with mean of `Items_qty_In_Grams` grouped by `pizza_name_id`
mean = ing.groupby('pizza_name_id')['Items_Qty_In_Grams'].mean()
ing['Items_Qty_In_Grams'] = ing['Items_Qty_In_Grams'].fillna(ing['pizza_name_id'].map(mean))
print(ing.info())

sale.to_csv('data/pizza_sale_cleaned.csv',index=False)
ing.to_csv('data/pizza_ingredients_cleaned.csv',index=False)


# data engineering
sale=pd.read_csv('data/pizza_sale_cleaned.csv')
ing=pd.read_csv('data/pizza_ingredients_cleaned.csv')

sale['pizza_name_id'] = sale['pizza_name_id'].str.lower()
print(sale)

sale['order_date'] = pd.to_datetime(sale['order_date'])
sale_agg = sale.groupby(['order_date', 'pizza_name_id','pizza_size','pizza_name','pizza_category','unit_price','total_price']).agg({'quantity': 'sum',}).reset_index()


# Calculate Z-scores for the 'quantity' column
sale_agg['z_score'] = (sale_agg['quantity'] - sale_agg['quantity'].mean()) / sale_agg['quantity'].std()

# Filter out outliers based on Z-score threshold of 3
threshold = 3
sale_filtered = sale_agg[np.abs(sale_agg['z_score']) <= threshold]
sale_outliers = sale_agg[np.abs(sale_agg['z_score']) > threshold]

# Drop the z_score column as it's no longer needed
sale_filtered = sale_filtered.drop(columns=['z_score'])
sale_outliers = sale_outliers.drop(columns=['z_score'])
sale_agg = sale_agg.drop(columns=['z_score'])

print(f"Original DataFrame shape: {sale_agg.shape}")
print(f"Outliers DataFrame shape: {sale_outliers.shape}")
print(f"Filtered DataFrame shape: {sale_filtered.shape}")

print(sale_agg.describe())
print(sale_outliers.describe())
print(sale_filtered.describe())
sale_filtered['order_date']=pd.to_datetime(sale_filtered['order_date'])
sale['order_date']=pd.to_datetime(sale['order_date'])
def extra_features_from_date(sale):
    sale['day_of_week'] = sale['order_date'].dt.weekday +1 #Starts with 1 as Monday to 7 as Sunday
    sale['day_of_year'] = sale['order_date'].dt.dayofyear 
    sale['day_of_month'] = sale['order_date'].dt.day
    sale['week_of_year'] = sale['order_date'].dt.strftime('%W') #week starts on Monday, new year days preceeding the first monday is week 0
    sale['month'] = sale['order_date'].dt.month    
    return sale
  
extra_features_from_date(sale_filtered)
  
import holidays
# using US holidays
us_holidays = holidays.US()

# Creating a 'holiday' column that indicates if the order_date was a holiday
sale_filtered['holiday'] = sale_filtered['order_date'].apply(lambda x: 1 if x in us_holidays else 0)

# Checking if the holiday flag works
sale_filtered[['order_date', 'holiday']].head()

# Creating a 'promotional_period' flag for weekends 
sale_filtered['promotion'] = sale_filtered['order_date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

# Checking if the promotion flag works
sale_filtered[['order_date', 'promotion']].head()
sale_filtered[sale_filtered['promotion'] == 1][['order_date', 'promotion']].head()

# Aggregateing sales data by date
daily_sales = sale_filtered.groupby('order_date')['quantity'].sum().reset_index()
# daily_sales.set_index('order_date', inplace=True)
daily_sales

print(sale_filtered)




# Ploting sales over time
sales_over_time = sale_filtered.groupby('order_date')['total_price'].sum().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=sales_over_time, x='order_date', y='total_price')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# Aggregate sales by pizza name
pizza_sales = sale_filtered.groupby('pizza_name')['quantity'].sum().reset_index()

# Sorting by quantity sold
pizza_sales = pizza_sales.sort_values(by='quantity', ascending=False)

# Ploting top 10 most popular pizzas
plt.figure(figsize=(12,6))
sns.barplot(data=pizza_sales.head(10), x='quantity', y='pizza_name', palette='viridis')
plt.title('Top 10 Most Popular Pizzas')
plt.xlabel('Quantity Sold')
plt.ylabel('Pizza Name')
plt.show()


# Ploting pizza size distribution
size_sales = sale_filtered.groupby('pizza_size')['quantity'].sum().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=size_sales, x='pizza_size', y='quantity', palette='coolwarm')
plt.title('Sales by Pizza Size')
plt.xlabel('Pizza Size')
plt.ylabel('Quantity Sold')
plt.show()


# Ploting pizza category distribution
category_sales = sale_filtered.groupby('pizza_category')['quantity'].sum().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=category_sales, x='pizza_category', y='quantity', palette='coolwarm')
plt.title('Sales by Pizza Category')
plt.xlabel('Pizza Category')
plt.ylabel('Quantity Sold')
plt.show()


# Aggregate sales by month
sales_by_month = sale_filtered.groupby('month')['total_price'].sum().reset_index()

# Plot sales by month
plt.figure(figsize=(10,5))
sns.barplot(data=sales_by_month, x='month', y='total_price', palette='magma')
plt.title('Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()


# Correlation matrix
corr_matrix = sale_filtered[['quantity', 'unit_price', 'total_price']].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()