import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product
import warnings
warnings.filterwarnings("ignore")
# input the training data (URPOP.xlsx/PCGDP.xlsx/SWD.xlsx)
data = pd.read_excel('URPOP.xlsx', sheet_name=0)


def output(x):
    x = pd.to_numeric(x)

    alpha_values = [0.01*i for i in range(101)]
    beta_values = [0.01*i for i in range(101)]
    best_r2 = 0
    best_params = (None, None)
    for alpha, beta in product(alpha_values, beta_values):
        model_holt = ExponentialSmoothing(x, trend='add')
        fitted_model = model_holt.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
        pop2 = fitted_model.fittedvalues
        r2 = r2_score(pop, pop2)
        if r2 > best_r2:
            best_r2 = r2
            best_params = (alpha, beta)
    # Output the optimal parameters
    print(f'optimal parameters: alpha = {np.round(best_params[0], decimals=2)}, '
          f'beta = {np.round(best_params[1], decimals=2)}, '
          f'R2 = {np.round(best_r2, decimals=2)}')
    # Train the final model with the best parameters and make predictions
    final_model = ExponentialSmoothing(x, trend='add').fit(smoothing_level=best_params[0],
                                                           smoothing_trend=best_params[1], optimized=False)
    # Predict the results for the remaining period from 2024 to 2060
    predictions = final_model.forecast(steps=37)
    return best_params[0], best_params[1], pd.to_numeric(predictions)


param = np.zeros((31, 2))
predict = np.zeros((31, 37))
for p in range(31):
    print(f'Start calculating region {p+1}')
    pop = data.loc[p]
    y = output(pop)
    param[p, 0], param[p, 1] = y[0], y[1]
    predict[p] = y[2].T

pd.DataFrame(predict).to_excel("predictions.xlsx")   # Output the prediction results to excel