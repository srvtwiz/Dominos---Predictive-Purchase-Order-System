# Dominos Predictive Purchase Order System

## Problem Statement
Dominos seeks to optimize ingredient ordering by accurately forecasting future pizza sales. By predicting demand, the company can ensure the right amount of ingredients is stocked, minimizing waste, reducing costs, and preventing stockouts. This project develops a predictive model that generates a purchase order based on forecasted sales and ingredient requirements.

---

## Business Use Cases

1. **Inventory Management**: Maintain optimal stock levels to meet demand.
2. **Cost Reduction**: Reduce waste and expenses associated with excess inventory.
3. **Sales Forecasting**: Inform promotional and business strategies using accurate predictions.
4. **Supply Chain Optimization**: Streamline ordering processes to align with demand and avoid disruptions.

---

## Data Sources

1. **Pizza Ingredients Dataset**:
   - File: `Pizza_ingredients.csv`
   - Contains details about pizza ingredients and quantities required per pizza.

2. **Pizza Sales Dataset**:
   - File: `pizza_sales.csv`
   - Historical sales data, including order dates, pizza types, sizes, and quantities.

---

## Approach

### 1. Data Preprocessing and Exploration

#### Steps:
- **Data Cleaning**:
  - Convert date formats.
  - Fill missing values using strategies such as mean imputation and mapping related data.
  - Remove or impute outliers using Z-scores.
- **Exploratory Data Analysis (EDA)**:
  - Identify trends, seasonality, and patterns in historical sales data.
  - Visualize features to gain insights for feature engineering.

#### Outputs:
- Cleaned sales and ingredients datasets.
- Visualizations showing seasonal trends and key patterns.

### 2. Sales Prediction

#### Steps:
- **Feature Engineering**:
  - Extract temporal features like day of the week, month, promotional periods, and holidays.
  - Create flags for promotional and holiday periods.
- **Model Selection**:
  - Evaluate models such as ARIMA, SARIMA, Prophet, and LSTM.
  - Choose the best model based on Mean Absolute Percentage Error (MAPE).
- **Model Training and Hyperparameter Tuning**:
  - Train models using historical data.
  - Perform grid search or manual tuning for optimal hyperparameters.

#### Outputs:
- Trained forecasting model.
- Evaluation results with metrics like MAPE.

### 3. Purchase Order Generation

#### Steps:
- **Sales Forecasting**:
  - Predict pizza sales for the next week.
- **Ingredient Calculation**:
  - Compute required ingredient quantities based on forecasted sales.
- **Generate Purchase Order**:
  - Produce a detailed purchase order listing ingredient quantities needed for the forecasted sales period.

#### Outputs:
- Predicted sales for the next week.
- Generated purchase order.

---

## Key Implementation Highlights

### Data Preprocessing
- **Handle Missing Values**:
  - Fill `total_price` using `quantity` and `unit_price`.
  - Map `pizza_ingredients` and `pizza_name_id` using existing mappings.
- **Remove Outliers**:
  - Use Z-scores to filter anomalies.

### Forecasting Models
- **LSTM**:
  - Applied for capturing non-linear dependencies and trends.
  - Trained with sequences generated from sales data.
- **Evaluation**:
  - MAPE used to compare model performance.

### Purchase Order Calculation
- Predicted ingredient quantities derived from pizza sales forecasts.
- Aggregate results to generate a consolidated order list.

---

## Results
- **Accuracy**: MAPE of the best-performing model.
- **Efficiency**: Reduced waste and stockouts in test scenarios.
- **Scalability**: Framework adaptable to multiple regions and product types.

---

## Example Outputs

### Sales Forecast
| Date       | Pizza Type | Predicted Quantity |
|------------|------------|--------------------|
| 2024-01-01 | Margherita | 120                |
| 2024-01-02 | Pepperoni  | 150                |

### Purchase Order
| Ingredient       | Required Quantity (g) |
|------------------|-----------------------|
| Mozzarella       | 10,000                |
| Tomato Sauce     | 5,000                 |

---

## Conclusion
The Dominos Predictive Purchase Order System provides an end-to-end solution for inventory optimization and sales forecasting. By leveraging data-driven insights, the system improves operational efficiency and supports strategic decision-making.

---



