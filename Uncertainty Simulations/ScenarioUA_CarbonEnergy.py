import numpy as np
import pandas as pd
from scipy.stats import triang

# Define the boundaries of the parameters and the parameters of the triangular distribution
# see Supplementary Table 27 for details
params = {
    'eR1': {'min': 118.94, 'mode': 148.68, 'max': 178.42},  # kWh t-1 wet sludge
    'eR2': {'min': 4.58, 'mode': 5.72, 'max': 6.86},  # kWh t-1 wet sludge
    'eR3': {'min': 394.66, 'mode': 493.33, 'max': 592},  # kWh t-1 wet sludge
    'cR1': {'min': 156.60, 'mode': 195.75, 'max': 234.90},  # kgCO2 t-1 wet sludge
    'cR2': {'min': 135.88, 'mode': 169.85, 'max': 203.82},  # kgCO2 t-1 wet sludge
    'cR3': {'min': 336.20, 'mode': 420.25, 'max': 504.30},  # kgCO2 t-1 wet sludge
    'cR4': {'min': 369.54, 'mode': 461.92, 'max': 554.30}  # kgCO2 t-1 wet sludge
}
# Monte Carlo simulation times
num_simulations = 1000


def Carbon_energy(sd, r, eR1, eR2, eR3, cR1, cR2, cR3, cR4):
    r1, r2, r3, r4 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    co2 = np.sum(sd * (cR1 * r1 + cR2 * r2 + cR3 * r3 + cR4 * r4) * 365 / 1e9)  # GHG emissions, Mt/year
    Power = np.sum(sd * (eR1 * r1 + eR2 * r2 + eR3 * r3) * 365 / 1e9)  # Energy recovery, TWh/year
    factor = co2 / Power
    return factor


def Output(sd, dR, eR1, eR2, eR3, cR1, cR2, cR3, cR4):
    rs = np.zeros((31, 4))  # The proportion of R1, R2, R3 and R4 routes from 2024 to 2060
    rs[:, 0] = dR['R1'].to_numpy()
    rs[:, 1] = dR['R2'].to_numpy()
    rs[:, 2] = dR['R3'].to_numpy()
    rs[:, 3] = dR['R4'].to_numpy()
    factor = np.zeros(num_simulations)
    for p in range(num_simulations):
        factor[p] = Carbon_energy(sd, rs, eR1[p], eR2[p], eR3[p], cR1[p], cR2[p], cR3[p], cR4[p])
    return factor


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

C_R1, C_R2, C_R3, C_R4 = sim_results['cR1'], sim_results['cR2'], sim_results['cR3'], sim_results['cR4']
E_R1, E_R2, E_R3 = sim_results['eR1'], sim_results['eR2'], sim_results['eR3']

# import data
df = pd.read_excel('Data.xlsx', sheet_name='Disposal')
dd = np.zeros((37, 31))
for i in range(37):
    dd[i] = df[2024 + i].to_numpy()  # Obtain the sludge disposal data for 2024 to 2060, t/day
# Obtain the proportion of the four routes in the specified scenario
dd_Rs = pd.read_excel('Data.xlsx', sheet_name='OPS1')
CE_future = np.zeros((37, num_simulations))
for i in range(37):
    CE_future[i] = Output(dd[i], dd_Rs, E_R1, E_R2, E_R3, C_R1, C_R2, C_R3, C_R4)
CE_future = pd.DataFrame(CE_future.T)
# Output the detailed results of 1000 Monte Carlo simulations to excel
CE_future.to_excel('CE_future.xlsx')
AvgCE_future = np.zeros((37, 2))
for i in range(37):
    AvgCE_future[i, 0] = np.mean(CE_future[i])
    AvgCE_future[i, 1] = np.std(CE_future[i])
# Output the mean and standard deviation of the Monte Carlo simulations to excel
pd.DataFrame(AvgCE_future).to_excel('AvgCE_future.xlsx')