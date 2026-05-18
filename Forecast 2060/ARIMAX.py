import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('SLD.xlsx').dropna(axis=1).set_index('Year')

endog = df['SLD']  # Define the dependent variable
exog = df[['SWD', 'URPOP', 'PCGDP']]  # Define independent variables

exog = sm.add_constant(exog)

train_end = len(df)
endog_train = endog.iloc[:train_end]
exog_train = exog.iloc[:train_end]

best_r2 = -np.inf
best_order = None
best_model = None

print("\nmodel selection...")

for p in range(6):
    for q in range(6):
        try:
            model = sm.tsa.SARIMAX(endog_train, exog=exog_train, order=(p, 0, q))
            results = model.fit(disp=False)
            fitted_values = results.fittedvalues
            r_squared = r2_score(endog_train, fitted_values)
            print(f"try model ARIMA({p},0,{q}) - R²: {r_squared:.4f}")
            if r_squared > best_r2:
                best_r2 = r_squared
                best_order = (p, 0, q)
                best_model = results
        except Exception as e:
            print(f"model ARIMA({p},0,{q}) cannot fit, error：{e}")
            continue

print(f"\noptimal model: ARIMA{best_order}，其 R² = {best_r2:.4f}")

if best_r2 >= 0.9:
    print("The model meets the requirement of R² >= 0.9")
else:
    print("The model failed to meet the requirement of R² >= 0.9. Please consider other models or obtain more data")

print("\nSummary of the Best Model：")
print(best_model.summary())

pre_pop = best_model.fittedvalues
future_df = pd.read_excel('SLD.xlsx', sheet_name=1).dropna(axis=1).set_index('Year')
future_exog = sm.add_constant(future_df)
predictions = best_model.predict(start=train_end, end=train_end + len(future_df) -1, exog=future_exog)
future_df['Predicted_SLD'] = predictions.values  # Future predicted value of SLD in 2024-2060

# Print the prediction results
print("\nFuture SLD prediction results (2024-2060):")
print(future_df[['SWD', 'URPOP', 'PCGDP', 'Predicted_SLD']])

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, endog, label='Actual values of the training set', marker='o')
plt.plot(df.index, pre_pop, label='Fitting values of the training set', marker='o')
plt.plot(future_df.index, future_df['Predicted_SLD'], label='Future predicted value', marker='o', linestyle='--', color='green')
plt.xlabel('Year')
plt.ylabel('SLD')
plt.title(f'Comparison of the SLD training set with future predicted values（ARIMA{best_order} model）')
plt.legend()
plt.grid(True)
plt.show()
future_df.to_excel('Predicted_SLD.xlsx')