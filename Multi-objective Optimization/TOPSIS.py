import numpy as np
import pandas as pd
data = pd.read_excel('Pareto_F.xlsx')
data.drop(columns=data.columns[0], axis=1, inplace=True)
data.columns = ['TAC', 'NGE', 'CEF']
data['TAC'] = 1 / data['TAC']    # TAC为极小型指标
data['NGE'] = 1 / data['NGE']    # net GHG emissions也为极小型指标
data['CEF'] = 1 / data['CEF']    # Carbon-energy也为极小型指标


def entropyWeight(dt):
    dt = np.array(dt)
    # Normalization
    P = dt / dt.sum(axis=0)
    # Calculate the entropy value
    E = np.nansum(-P * np.log(P) / np.log(len(dt)), axis=0)
    # Calculate the weight coefficient
    return (1 - E) / (1 - E).sum()


def topsis(dt, weight=None):
    # Normalization
    dt = dt / np.sqrt((dt ** 2).sum())

    # The best and worst solution
    Z = pd.DataFrame([dt.min(), dt.max()], index=['Negative ideal solution', 'Positive ideal solution'])

    # Distance
    weight = entropyWeight(dt) if weight is None else np.array(weight)
    Result = dt.copy()
    Result['Positive ideal solution'] = np.sqrt(((dt - Z.loc['Positive ideal solution']) ** 2 * weight).sum(axis=1))
    Result['Negative ideal solution'] = np.sqrt(((dt - Z.loc['Negative ideal solution']) ** 2 * weight).sum(axis=1))

    # Composite score index
    Result['Composite score index'] = Result['Negative ideal solution'] / (Result['Negative ideal solution'] + Result['Positive ideal solution'])
    Result['Sort'] = Result.rank(ascending=False)['Composite score index']

    return Result


Results = topsis(data, weight=[1/3, 1/3, 1/3])  # The weights set for the three objectives are the same
Results.to_excel('TOPSIS.xlsx')  # Save the data after the topsis decision