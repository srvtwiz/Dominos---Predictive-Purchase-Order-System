import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from itertools import product
import matplotlib.pyplot as plt
from warnings import filterwarnings
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

sale=pd.read_csv('data/pizza_sale_cleaned.csv')
ing=pd.read_csv('data/pizza_ingredients_cleaned.csv')

sale['pizza_name_id'] = sale['pizza_name_id'].str.lower()
print(sale)

import numpy as np

sale['order_date'] = pd.to_datetime(sale['order_date'])
sale_agg = sale.groupby(['order_date', 'pizza_name_id','pizza_size','pizza_name','pizza_category','unit_price']).agg({'quantity': 'sum',}).reset_index()


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

# sale_filtered.drop(['order_date','pizza_name_id'],axis=1,inplace=True)
# print(sale_filtered.dtypes)

# sale_agg['order_date'] = pd.to_datetime(sale_agg['order_date'], format='%Y-%m-%d')

# data = pd.pivot_table(
#            data    = sale_agg,
#            values  = 'quantity',
#            index   = 'order_date',
#            columns = 'pizza_name_id'
#        )
# data.columns.name = None
# data.columns = [f"{col}" for col in data.columns]
# data = data.asfreq('1D')
# data = data.sort_index()
# data.head(4)

# data.fillna(0, inplace=True)
# data.head(4)


# import matplotlib.pyplot as plt
# import skforecast
# from skforecast.plot import set_dark_theme

# # Plot time series for first 4 pizzas alone 
# # ======================================================================================
# set_dark_theme()
# fig, axs = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
# data.iloc[:, :4].plot(
#     legend   = True,
#     subplots = True, 
#     title    = 'Pizza Quantities',
#     ax       = axs, 
# )
# fig.tight_layout()
# plt.show()

# Split data into training and testing sets
train_data = sale_filtered[:-30]  # All but the last 30 days
test_data = sale_filtered[-30:]   # Last 30 days

# Normalize the data for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Prepare sequences for LSTM
sequence_length = 7
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=sequence_length, batch_size=1)
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=sequence_length, batch_size=1)

# Function to calculate MAPE
def evaluate_model(true, predicted):
    return mean_absolute_percentage_error(true, predicted)
  
def train_arima(train, test):
    best_model = None
    best_mape = float('inf')
    best_params = None

    # Hyperparameter grid
    p = d = q = range(0, 3)
    pdq = list(product(p, d, q))

    for params in pdq:
        try:
            model = ARIMA(train, order=params).fit()
            predictions = model.forecast(steps=len(test))
            mape = evaluate_model(test, predictions)
            if mape < best_mape:
                best_mape = mape
                best_model = model
                best_params = params
        except:
            continue

    print(f"Best ARIMA Params: {best_params}, MAPE: {best_mape:.2%}")
    return best_model

arima_model = train_arima(train_data['quantity'], test_data['quantity'])

# 2. SARIMA Model
def train_sarima(train, test):
    best_model = None
    best_mape = float('inf')
    best_params = None

    # Hyperparameter grid
    p = d = q = range(0, 3)
    seasonal_pdq = [(x[0], x[1], x[2], 7) for x in product(p, d, q)]
    
    for params in product(p, d, q):
        for seasonal_params in seasonal_pdq:
            try:
                model = SARIMAX(train, order=params, seasonal_order=seasonal_params).fit()
                predictions = model.forecast(steps=len(test))
                mape = evaluate_model(test, predictions)
                if mape < best_mape:
                    best_mape = mape
                    best_model = model
                    best_params = (params, seasonal_params)
            except:
                continue

    print(f"Best SARIMA Params: {best_params}, MAPE: {best_mape:.2%}")
    return best_model
  
sarima_model = train_sarima(train_data['quantity'], test_data['quantity'])

# 4. LSTM Model
def train_lstm(train_gen, test_gen):
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_gen, epochs=20, verbose=1)
    predictions = model.predict(test_gen)
    predictions = scaler.inverse_transform(predictions)
    true = scaler.inverse_transform([test_scaled[sequence_length:]])
    mape = evaluate_model(true.flatten(), predictions.flatten())
    print(f"LSTM MAPE: {mape:.2%}")
    return model

lstm_model = train_lstm(train_generator, test_generator)
  
  # Compare Models
print("\nModel Comparison:")
print(f"ARIMA MAPE: {evaluate_model(test_data['quantity'], arima_model.forecast(steps=30)):.2%}")
print(f"SARIMA MAPE: {evaluate_model(test_data['quantity'], sarima_model.forecast(steps=30)):.2%}")
print(f"LSTM MAPE: {evaluate_model(test_data['quantity'], lstm_model.predict(test_generator)):.2%}")