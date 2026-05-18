import numpy as np
import pandas as pd
from scipy.stats import triang

# Define the boundaries of the parameters and the parameters of the triangular distribution
# see Supplementary Table 27 for details
params = {
    'eR1': {'min': 127.97, 'mode': 159.96, 'max': 191.95},  # kWh t-1 Sludge
    'eR2': {'min': 21.02, 'mode': 26.28, 'max': 31.54},  # kWh t-1 Sludge
    'eR3': {'min': 119.95, 'mode': 149.94, 'max': 179.93},  # kWh t-1 Sludge
    'eR4': {'min': 31.20, 'mode': 39, 'max': 46.80},  # kWh t-1 Sludge
    'cR1': {'min': 50, 'mode': 55, 'max': 60},  # 10,000 CNY t-1 Sludge/day
    'cR2': {'min': 30, 'mode': 40, 'max': 50},  # 10,000 CNY t-1 Sludge/day
    'cR3': {'min': 30, 'mode': 40, 'max': 50},  # 10,000 CNY t-1 Sludge/day
    'cR4': {'min': 15, 'mode': 20, 'max': 25},  # 10,000 CNY t-1 Sludge/day
    'ele': {'min': 0.4108, 'mode': 0.5135, 'max': 0.6162}  # electricity price, CNY/kWh
}
# Monte Carlo simulation times
num_simulations = 1000


def TAC(sd, dR, eR1, eR2, eR3, eR4, cR1, cR2, cR3, cR4, ele, C0, D0):
    r1, r2, r3, r4 = dR['R1'].to_numpy(), dR['R2'].to_numpy(), dR['R3'].to_numpy(), dR['R4'].to_numpy()
    C1, C2, C3, C4 = C0  # The fixed investment costs of routes R1, R2, R3 and R4 in the previous year
    D1, D2, D3, D4 = D0  # The processing capacity t/d of routes R1, R2, R3 and R4 in the previous year
    # The processing volume t/d of R1, R2, R3 and R4 in a future year under the specified scenario
    F1, F2, F3, F4 = sd * r1, sd * r2, sd * r3, sd * r4

    cop = (eR1 * F1 + eR2 * F2 + eR3 * F3 + eR4 * F4) * 365 * ele / 1e6  # million CNY/y
    cap_r1 = np.copy(C1)
    cap_r2 = np.copy(C2)
    cap_r3 = np.copy(C3)
    cap_r4 = np.copy(C4)

    # Calculate the investment cost of Route 1
    c1_r1 = F1 > D1
    c2_r1 = (D1==0)
    a_r1 = c1_r1 & c2_r1
    b_r1 = c1_r1 & ~c2_r1
    cap_r1[a_r1] = cR1 * F1[a_r1] / 100  # million CNY/y
    cap_r1[b_r1] = C1[b_r1] * (F1[b_r1] / D1[b_r1]) ** 0.6
    cap_r1[~c1_r1] = 0
    cap_rr1 = np.copy(cap_r1)
    cap_rr1[~c1_r1] = C1[~c1_r1]
    F1[~c1_r1] = D1[~c1_r1]

    # Calculate the investment cost of Route 2
    c1_r2 = F2 > D2
    c2_r2 = (D2==0)
    a_r2 = c1_r2 & c2_r2
    b_r2 = c1_r2 & ~c2_r2
    cap_r2[a_r2] = cR2 * F2[a_r2] / 100  # million CNY/y
    cap_r2[b_r2] = C2[b_r2] * (F2[b_r2] / D2[b_r2]) ** 0.6
    cap_r2[~c1_r2] = 0
    cap_rr2 = np.copy(cap_r2)
    cap_rr2[~c1_r2] = C2[~c1_r2]
    F2[~c1_r2] = D2[~c1_r2]

    # Calculate the investment cost of Route 3
    c1_r3 = F3 > D3
    c2_r3 = (D3==0)
    a_r3 = c1_r3 & c2_r3
    b_r3 = c1_r3 & ~c2_r3
    cap_r3[a_r3] = cR3 * F3[a_r3] / 100  # million CNY/y
    cap_r3[b_r3] = C3[b_r3] * (F3[b_r3] / D3[b_r3]) ** 0.6
    cap_r3[~c1_r3] = 0
    cap_rr3 = np.copy(cap_r3)
    cap_rr3[~c1_r3] = C3[~c1_r3]
    F3[~c1_r3] = D3[~c1_r3]

    # Calculate the investment cost of Route 4
    c1_r4 = F4 > D4
    c2_r4 = (D4==0)
    a_r4 = c1_r4 & c2_r4
    b_r4 = c1_r4 & ~c2_r4
    cap_r4[a_r4] = cR4 * F4[a_r4] / 100  # million CNY/y
    cap_r4[b_r4] = C4[b_r4] * (F4[b_r4] / D4[b_r4]) ** 0.6
    cap_r4[~c1_r4] = 0
    cap_rr4= np.copy(cap_r4)
    cap_rr4[~c1_r4] = C4[~c1_r4]
    F4[~c1_r4] = D4[~c1_r4]

    cap = np.vstack([cap_rr1, cap_rr2, cap_rr3, cap_rr4])
    F = np.vstack([F1, F2, F3, F4])

    tac = cop + cap_r1 + cap_r2 + cap_r3 + cap_r4

    return tac * 0.14191 / 1000, cap, F   # total annual costs, billion US$


def Output1(sd, dR, eR1, eR2, eR3, eR4, cR1, cR2, cR3, cR4, ele, C0, D0):
    t1 = np.zeros((37, 31))
    for p in range(37):
        t1[p, :], C0, D0 = TAC(sd[p, :], dR, eR1, eR2, eR3, eR4, cR1, cR2, cR3, cR4, ele, C0, D0)
    return np.sum(t1, axis=1)


def Output2(sd, dR, eR1, eR2, eR3, eR4, cR1, cR2, cR3, cR4, ele, C0, D0):
    t2 = np.zeros((num_simulations, 37))
    for p in range(num_simulations):
        t2[p, :] = Output1(sd, dR, eR1[p], eR2[p], eR3[p], eR4[p], cR1[p], cR2[p], cR3[p], cR4[p], ele[p], C0, D0)
    return t2


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
E_R1, E_R2, E_R3, E_R4, Ele = sim_results['eR1'], sim_results['eR2'], sim_results['eR3'], sim_results['eR4'], sim_results['ele']

# import data
df = pd.read_excel('Data.xlsx', sheet_name='Disposal')
df2023 = pd.read_excel('Data.xlsx', sheet_name='Basis2023')
dr = pd.read_excel('Data.xlsx',sheet_name='OPS1')
dd = np.zeros((37, 31))  # Obtain the sludge disposal data for 2024 to 2060, t/day
C = np.zeros((4, 31))  # Obtain the fixed investment cost data of R1, R2, R3 and R4 routes in 2023
D = np.zeros((4, 31))  # Obtain the sludge treatment capacity (t/d) of routes R1, R2, R3 and R4 in 2023
for i in range(37):
    dd[i] = df[2024 + i].to_numpy()  # Obtain the sludge disposal data for 2024 to 2060, t/day
for i in range(4):
    C[i] = df2023[i + 1].to_numpy()  # Obtain the fixed investment cost data of R1, R2, R3 and R4 routes in 2023
    D[i] = df2023[i + 5].to_numpy()  # Obtain the sludge treatment capacity (t/d) of routes R1, R2, R3 and R4 in 2023
Tt_future = Output2(dd, dr, E_R1, E_R2, E_R3, E_R4, C_R1, C_R2, C_R3, C_R4, Ele, C, D)
Tt_future = pd.DataFrame(Tt_future)
# Output the detailed results of 1000 Monte Carlo simulations to excel
Tt_future.to_excel('Tt_future.xlsx')
AvgTt_future = np.zeros((37, 2))
for i in range(37):
    AvgTt_future[i, 0] = np.mean(Tt_future[i])
    AvgTt_future[i, 1] = np.std(Tt_future[i])
# Output the mean and standard deviation of the Monte Carlo simulations to excel
pd.DataFrame(AvgTt_future).to_excel('AvgTt_future.xlsx')