
import xgboost
import shap
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
import seaborn as sns
import scipy


## =========================== Part 1: Loading Data ===========================

start= time.time()
input1 = pd.read_csv("inputdata.csv", names = [r'$b/d_{50}$',r'$y/b$',r'$Fr$', r'$V/V_c$'])
output1 = pd.read_csv("outputdata.csv", names = [r'$d_s/y$'])


# dataset loading
value_input = input1.values
value_output = output1.values

shaped_input = value_input.reshape(528,4)
shaped_output = value_output.reshape(528, 1)


## ====================== Part 2: Creating Train and Test Sets ======================

X_train, X_test, y_train, y_test = train_test_split(shaped_input, shaped_output ,test_size=0.2, random_state=42)



## ====================== Part 3: Defining XGBoost Model Parameters and Training XGBoost Model ======================
params = {'objective': 'reg:squarederror',
          'base_score': 1,
          'booster': 'gbtree',
          'colsample_bylevel': 1,
          'colsample_bytree': 1,
          'n_estimators': 800,
          'learning_rate': 0.03,
          'gamma':0,
          'subsample':0.7,
          'max_depth':6,
          'min_child_weight':8,
          'scale_pos_weight':1,
          'reg_alpha': 0.1,
          }

xgb_model = xgboost.XGBRegressor(**params)

print(len(X_train), len(X_test))
xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='rmse',  eval_set=[(X_test, y_test)], verbose=False)
mse = mean_squared_error(y_test, xgb_model.predict(X_test))


## ====================== Part 41: Using model ======================

bd50_values = []
y_over_b_values = []
Fr_values = []
V_over_Vc_values = []

# Get user input for each feature
num_inputs = int(input("Enter the number of predictions you want to make: "))
for _ in range(num_inputs):
    bd50_values.append(float(input("Enter $b/d_{50}$ value: ")))
    y_over_b_values.append(float(input("Enter $y/b$ value: ")))
    Fr_values.append(float(input("Enter $Fr$ value: ")))
    V_over_Vc_values.append(float(input("Enter $V/V_c$ value: ")))

# Create a dictionary with the collected input data
input_data = {
    '$b/d_{50}$': bd50_values,
    '$y/b$': y_over_b_values,
    '$Fr$': Fr_values,
    '$V/V_c$': V_over_Vc_values
}

# Create a DataFrame from the input data
input_df = pd.DataFrame.from_dict(input_data)

# Make predictions using the loaded model
predicted_values = xgb_model.predict(input_df.values)
Revised_predicted_value= predicted_values*1.3

# Display the predicted values
for i, predicted_value in enumerate(predicted_values):
    print(f"Predicted Value {i + 1}: {predicted_value}")
for i, predicted_value in enumerate(predicted_values):
    print(f"Revised Predicted Value {i + 1}: {Revised_predicted_value}")


explainer = shap.Explainer(xgb_model, input1)
shap_values = explainer(input_df.values)
shap.plots.waterfall(shap_values[0])
