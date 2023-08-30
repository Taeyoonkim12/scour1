
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
xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='rmse',  eval_set=[(X_test, y_test)], verbose=False)
mse = mean_squared_error(y_test, xgb_model.predict(X_test))


## ====================== Part 4: Metrics and Analysis ======================
Y_prediction_test= xgb_model.predict(X_test) ## Prediction of Test set
Y_prediction_train = xgb_model.predict(X_train) ## Prediction of Train set
Y_total = xgb_model.predict(input1) ## Prediction of Totall

print("time :", time.time() - start)

test = y_test ## Measured value(Test set)
prediction_test = Y_prediction_test ## Predicted value (Test set)
train = y_train ## Measured value (Training set)
prediction_train = Y_prediction_train ## Predicted value (Training set)


test = test.flatten()
prediction_test = prediction_test.flatten()
train = train.flatten()
prediction_train = prediction_train.flatten()

Mean_test = np.mean(test)
Mean_prediction_test = np.mean(prediction_test)
Mean_train = np.mean(prediction_train)
Mean_prediction_train = np.mean(train)



## RMSE
RMSE_test = math.sqrt(sum([(prediction_test[i]-test[i])**2/len(test) for i in range(len(test))]))
print("RMSE_test: %f" % RMSE_test)
RMSE_train = math.sqrt(sum([(train[i]-prediction_train[i])**2/len(train) for i in range(len(train))]))
print("RMSE_train: %f" % RMSE_train)

## NMSE
NMSE_test = sum([(prediction_test[i]-test[i])**2/(Mean_test*Mean_prediction_test*len(test)) for i in range(len(test))])
print("NMSE_test: %f" % NMSE_test)
NMSE_train = math.sqrt(sum([(train[i]-prediction_train[i])**2/(Mean_train*Mean_prediction_train*len(train)) for i in range(len(train))]))
print("NMSE_train: %f" % NMSE_train)

## I
divisor = sum([(prediction_test[i]-test[i])**2 for i in range(len(test))])
dividend = sum([(abs(test[i]-Mean_test)+abs(prediction_test[i]-Mean_test))**2 for i in range(len(test))])
Ia_test= 1-divisor/dividend
print("Itest: %f" % Ia_test)

divisor = sum([(prediction_train[i]-train[i]) **2 for i in range(len(train))])
dividend = sum([(abs(train[i]-Mean_train)+abs(prediction_train[i]-Mean_train))**2 for i in range(len(train))])
Ia_train= 1-divisor/dividend
print("Itrain: %f" % Ia_train)

## SI
divisor = math.sqrt(sum([(prediction_test[i]-test[i])**2/len(test) for i in range(len(test))]))
SI_test = divisor/Mean_test
print("SI_test: %f" % SI_test)
divisor = math.sqrt(sum([(train[i]-prediction_train[i])**2/len(train) for i in range(len(train))]))
SI_train = divisor/Mean_prediction_train
print("SI_train: %f" % SI_train)

## NSE
divisor = sum([(test[i]-prediction_test[i]) **2 for i in range(len(test))])
dividend = sum([(test[i]-Mean_test) **2 for i in range(len(test))])
nse_test= 1-divisor/dividend
print("nse_test: %f" % nse_test)

divisor = sum([(prediction_train[i]-train[i]) **2 for i in range(len(train))])
dividend = sum([(train[i]-Mean_prediction_train) **2 for i in range(len(train))])
nse_train= 1-divisor/dividend
print("nse_train: %f" % nse_train)

## R2
a1 = sum([(test[i]-Mean_test)*(prediction_test[i]-Mean_prediction_test) for i in range(len(test))])
a2 = math.sqrt(sum([(test[i]-Mean_test)**2 for i in range(len(test))]))
a3 = math.sqrt(sum([(prediction_test[i]-Mean_prediction_test)**2 for i in range(len(test))]))
r2_test= (a1/(a2*a3))**2
print("r_test: %f" % r2_test)

a1 = sum([(train[i]-Mean_train)*(prediction_train[i]-Mean_prediction_train) for i in range(len(train))])
a2 = math.sqrt(sum([(train[i]-Mean_train)**2 for i in range(len(train))]))
a3 = math.sqrt(sum([(prediction_train[i]-Mean_prediction_train)**2 for i in range(len(train))]))
r2_train= (a1/(a2*a3))**2
print("r_train: %f" % r2_train)


## B
B_test = sum([(prediction_test[i]-test[i])/len(test) for i in range(len(test))])
print("B_test: %f" % B_test)

B_train = sum([(prediction_train[i]-train[i])/len(prediction_train) for i in range(len(prediction_train))])
print("B_train: %f" % B_train)

## Se
Se_test = math.sqrt(sum([((prediction_test[i]-test[i])-B_test)**2/(len(test)-2) for i in range(len(test))]))
print("SE_test: %f" % Se_test)
Se = math.sqrt(sum([((prediction_train[i]-train[i])-B_train)**2/(len(prediction_train)-2) for i in range(len(prediction_train))]))
print("SE_train: %f" % Se)


# Measured versus Predicted
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Test
axs[0].plot(test, prediction_test, 'o', label='Predicted Data', markersize=8, alpha=0.7)
axs[0].plot(test, test, color='red', linestyle='dashed', label='1:1 Line', linewidth=2)
axs[0].set_xlabel('Oveserved Values', fontsize=12)
axs[0].set_ylabel('Predicted Values', fontsize=12)
axs[0].set_title('Test Data Prediction', fontsize=14)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)

# Training
axs[1].plot(train, prediction_train, 'o', label='Predicted Data', markersize=8, alpha=0.7)
axs[1].plot(train, train, color='red', linestyle='dashed', label='1:1 Line', linewidth=2)
axs[1].set_xlabel('Oveserved Values', fontsize=12)
axs[1].set_ylabel('Predicted Values', fontsize=12)
axs[1].set_title('Train Data Prediction', fontsize=14)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()  # Adjust layout for better spacing

# Show the subplots
plt.show()

# Error distribution
Error_test = prediction_test- test
Error_train = prediction_train- train


## Error distribution

# Error distribution for Test set
fig = plt.figure(figsize=(20, 10))
plt.subplot(121)
sns.distplot(
    Error_test,
    kde=False,
    fit=scipy.stats.norm,
    bins=15,
    hist_kws={
        "rwidth": 0.8,
        "color": "blue",
        "edgecolor": "black",
        "alpha": 0.7
    },
    fit_kws={
        "color": "red",
        "linestyle": "dashed"
    }
)
plt.xlim(-1.5, 1.5)
plt.xlabel('$y_{measured} - y_{predicted}$')
plt.ylabel("Normalized PDF")
plt.title("Error Distribution (Test Set)")
plt.grid(True)

# Error distribution for Training set
plt.subplot(122)  # Change the subplot number to 122
sns.distplot(
    Error_train,
    kde=False,
    fit=scipy.stats.norm,
    bins=15,
    hist_kws={
        "rwidth": 0.8,
        "color": "blue",
        "edgecolor": "black",
        "alpha": 0.7
    },
    fit_kws={
        "color": "red",
        "linestyle": "dashed"
    }
)
plt.xlim(-1.5, 1.5)
plt.xlabel('$y_{measured} - y_{predicted}$')
plt.ylabel("Normalized PDF")
plt.title("Error Distribution (Training Set)")
plt.grid(True)

plt.tight_layout()
plt.show()


## ====================== Part 5: Model Generalization ======================

#Johnes+sheppard
regulation= pd.read_csv("regulation_renew.csv", names = [r'$b/d_50$',r'$y/b$',r'$Fr$', r'$V/V_c$', r' $y_s/b$'])
X_regulation= regulation.loc [:, r'$b/d_50$': r'$V/V_c$']
Y_regulation_observed = regulation.loc [:,r' $y_s/b$']
Y_regulation_predicted = xgb_model.predict(X_regulation)


# Generalization Figure
X=np.linspace(0,3.5,100)
y1=X+ B_test
y2=X+(B_test+1.96*Se_test)
y3=X-(B_test+1.96*Se_test)

plt.figure(figsize=(10, 6))
plt.scatter(Y_regulation_observed, Y_regulation_predicted, color='blue', alpha=0.7)
plt.plot(X, y1, color='red', linestyle='solid')
plt.plot(X, y2, color='red', linestyle='dashed')
plt.plot(X, y3, color='red', linestyle='dotted')
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)
plt.xlabel('Observed value($y_{so/b}$)', fontsize=12)
plt.ylabel('Predicted value($y_{sp/b}$)', fontsize=12)
plt.title('Observed vs. Predicted Values ', fontsize=14)
plt.legend(['Genralization', 'Mean Line of Prediction Error','Upper Line of 95% C.I.', 'Lower Line of 95% C.I.'])
plt.grid(True)
plt.tight_layout()
plt.show()


## ====================== Part 6: Model Interpretation and Visualization ======================

# Initialize JavaScript for SHAP plot
shap.initjs()
# Create a SHAP TreeExplainer
explainer = shap.TreeExplainer(xgb_model)
# Calculate SHAP values
shap_values = explainer.shap_values(input1)

# Force plot for a single prediction
shap.force_plot(explainer.expected_value, shap_values, input1)

# Summary plot for feature importance
shap.summary_plot(shap_values, input1)
shap.summary_plot(shap_values, input1, plot_type="bar", color='blue')

# Individual feature dependence plots
for feature_idx in range(input1.shape[1]):
    shap.dependence_plot(feature_idx, shap_values, input1, dot_size=45)

# Individual feature dependence plots without interaction_index
for feature_idx in range(input1.shape[1]):
    shap.dependence_plot(feature_idx, shap_values, input1, interaction_index=None, dot_size=45)



# Example waterfall plots for specific cases
explainer = shap.Explainer(xgb_model, input1)
shap_values = explainer(input1)
shap.plots.heatmap(shap_values)
shap.plots.waterfall(shap_values[261]) ##Case 1  b/d_50=3.67, V/V_c=0.95, y/b =20.95, Fr =0.50
shap.plots.waterfall(shap_values[385]) ##Case 2  b/d50=203.6, V/V_c=3.91, y/b =2.0, Fr=1
shap.plots.waterfall(shap_values[359]) ##Case 3-1 b'=0.33, V=1.15, y=1.97, d50=0.55, sediment nonuniformity= 4.6, ysm/b=0.45
shap.plots.waterfall(shap_values[362]) ##Case 3-2 b'=0.33, V=1.15, y=1.97, d50=0.55, sediment nonuniformity= 1.6, ysm/b=2.20
shap.plots.waterfall(shap_values[363]) ##Case 4-1 b'=0.33, V=1.38, y=1.97, d50=0.85, sediment nonuniformity= 3.3, ysm/b=0.40
shap.plots.waterfall(shap_values[367]) ##Case 4-2 b'=0.33, V=1.15, y=1.97, d50=0.85, sediment nonuniformity= 1.3, ysm/b=2.10
shap.plots.waterfall(shap_values[368]) ##Case 5-1 b'=0.33, V=1.15, y=1.97, d50=0.85, sediment nonuniformity= 1.3, ysm/b=2.10


# Force plot for a single prediction
shap.initjs()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(input1)
shap.force_plot(explainer.expected_value, shap_values[261], input1.iloc[261, :], matplotlib=True)



## ====================== Part 7: Using model ======================

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
