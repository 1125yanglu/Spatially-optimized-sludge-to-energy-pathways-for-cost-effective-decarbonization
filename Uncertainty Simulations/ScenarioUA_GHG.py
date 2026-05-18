import numpy as np
import pandas as pd
from scipy.stats import triang

# Define the boundaries of the parameters and the parameters of the triangular distribution
# see Supplementary Table 27 for details
params = {
    'R1': {'min': 156.60, 'mode': 195.75, 'max': 234.90},  # kgCO2 t-1 wet sludge (80% moisture)
    'R2': {'min': 135.88, 'mode': 169.85, 'max': 203.82},  # kgCO2 t-1 wet sludge (80% moisture)
    'R3': {'min': 336.20, 'mode': 420.25, 'max': 504.30},  # kgCO2 t-1 wet sludge (80% moisture)
    'R4': {'min': 369.54, 'mode': 461.92, 'max': 554.30}   # kgCO2 t-1 wet sludge (80% moisture)
}
# Monte Carlo simulation times
num_simulations = 1000


def Carbon(sd, r, E1, E2, E3, E4):
    r1, r2, r3, r4 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    co2 = sd * (E1 * r1 + E2 * r2 + E3 * r3 + E4 * r4) * 365 / 1e9  # GHG emissions, MtCO2eq
    return co2


def Output(SD, dd_rs, E1, E2, E3, E4):
    CO2 = np.zeros(num_simulations)
    rs = np.zeros((31, 4))  # The proportion of R1, R2, R3 and R4 routes from 2024 to 2060
    rs[:, 0] = dd_rs['R1'].to_numpy()
    rs[:, 1] = dd_rs['R2'].to_numpy()
    rs[:, 2] = dd_rs['R3'].to_numpy()
    rs[:, 3] = dd_rs['R4'].to_numpy()

    for p in range(num_simulations):
        CO2[p] = np.sum(Carbon(SD, rs, E1[p], E2[p], E3[p], E4[p]))

    return CO2


# Store the value of each parameter in all simulations
sim_results = {key: [] for key in params.keys()}

# For each parameter, a Monte Carlo simulation is conducted using a triangular distribution
for param, distribution in params.items():
    # Calculate the parameters of the triangular distribution
    c = (distribution['mode'] - distribution['min']) / (distribution['max'] - distribution['min'])
    # Generate samples
    samples = triang.rvs(c, loc=distribution['min'], scale=distribution['max'] - distribution['min'], size=num_simulations)
    # Store the results
    sim_results[param] = samples

C_R1, C_R2, C_R3, C_R4 = sim_results['R1'], sim_results['R2'], sim_results['R3'], sim_results['R4']

# import data
df = pd.read_excel('Data.xlsx', sheet_name='Disposal')  # sludge disposal t/day
dd = np.zeros((37, 31))
for i in range(37):
    dd[i] = df[2024 + i].to_numpy()  # Obtain the sludge disposal data for 2024 to 2060, t/day
# Obtain the proportion of the four routes in the specified scenario
dd_Rs = pd.read_excel('Data.xlsx', sheet_name='OPS1')
Ct_future = np.zeros((37, num_simulations))
for i in range(37):
    Ct_future[i] = Output(dd[i], dd_Rs, C_R1, C_R2, C_R3, C_R4)
Ct_future = pd.DataFrame(Ct_future.T)
# Output the detailed results of 1000 Monte Carlo simulations to excel
Ct_future.to_excel('Ct_future.xlsx')
AvgCt_future = np.zeros((37, 2))
for i in range(37):
    AvgCt_future[i, 0] = np.mean(Ct_future[i])
    AvgCt_future[i, 1] = np.std(Ct_future[i])
# Output the mean and standard deviation of the Monte Carlo simulations to excel
pd.DataFrame(AvgCt_future).to_excel('AvgCt_future.xlsx')