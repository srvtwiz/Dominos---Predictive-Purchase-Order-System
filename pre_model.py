import pandas as pd
import matplotlib.pyplot as plt

# Load the data
ing=pd.read_csv('data/Pizza_ingredients - Pizza_ingredients.csv')
sale=pd.read_csv('data/Pizza_Sale - pizza_sales.csv')

print(sale.info())
print(ing.info())

# Combine `order_date` and `order_time` into a single string
sale['datetime_string'] = sale['order_date'] + ' ' + sale['order_time']

# First attempt parsing using the default format
sale['order_datetime'] = pd.to_datetime(sale['datetime_string'], errors='coerce')

# Handle rows that could not be parsed
invalid_dates = sale[sale['order_datetime'].isnull()]

if not invalid_dates.empty:
    print(f"Rows with invalid datetime format: {len(invalid_dates)}")
    
    # Retry parsing with alternative format for unparsed rows
    sale.loc[sale['order_datetime'].isnull(), 'order_datetime'] = pd.to_datetime(
        invalid_dates['datetime_string'], format='%d-%m-%Y %H:%M:%S', errors='coerce'
    )

# Verify if all rows are now parsed
final_invalid = sale[sale['order_datetime'].isnull()]
if not final_invalid.empty:
    print(f"Still unparsed rows: {len(final_invalid)}")
    print(final_invalid)

# Drop intermediate column if not needed
sale.drop(columns=['order_date', 'order_time', 'datetime_string'], inplace=True)

#handling missing values
sale['total_price']=sale['total_price'].fillna(sale['quantity']*sale['unit_price'])

# Fill missing `pizza_ingredients` in sale by `pizza_name`
def fill_pizza_ingredients(row, ing):
    if pd.isnull(row['pizza_ingredients']) and pd.notnull(row['pizza_name']):
        match = ing[ing['pizza_name'] == row['pizza_name']]
        if not match.empty:
            return match['pizza_ingredients'].iloc[0]
    return row['pizza_ingredients']

sale['pizza_ingredients'] = sale.apply(fill_pizza_ingredients, axis=1, ing=ing)

# Fill missing `pizza_name_id` in sale by comparing `pizza_ingredients`
def fill_pizza_name_id(row, ing):
    if pd.isnull(row['pizza_name_id']) and pd.notnull(row['pizza_name']):
        match = ing[ing['pizza_name'] == row['pizza_name']]
        if not match.empty:
            return match['pizza_name_id'].iloc[0]
    return row['pizza_name_id']

sale['pizza_name_id'] = sale.apply(fill_pizza_name_id, axis=1, ing=ing)


# Fill missing `pizza_name` in sales_df by `pizza_name_id`
def fill_pizza_name(row, ing):
    if pd.isnull(row['pizza_name']) and pd.notnull(row['pizza_name_id']):
        match = ing[ing['pizza_name_id'] == row['pizza_name_id']]
        if not match.empty:
            return match['pizza_name'].iloc[0]
    return row['pizza_name']

sale['pizza_name'] = sale.apply(fill_pizza_name, axis=1, ing=ing)


#Create a mapping of pizza_name to its most frequent pizza_category
category_mapping = (sale.groupby("pizza_name_id")["pizza_category"].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict())

# a function to fill missing pizza_category values
def fill_pizza_category(row, category_mapping):
    if pd.isnull(row["pizza_category"]):
        return category_mapping.get(row["pizza_name_id"], None)
    return row["pizza_category"]

sale["pizza_category"] = sale.apply(fill_pizza_category, axis=1, category_mapping=category_mapping)

print(sale.info())




print(ing.info())