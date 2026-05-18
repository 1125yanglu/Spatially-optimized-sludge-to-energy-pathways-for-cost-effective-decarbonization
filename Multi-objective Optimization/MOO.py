# import the required library and data
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
df = pd.read_excel('Data.xlsx', sheet_name='Disposal')
ds = pd.read_excel('Data.xlsx', sheet_name='Basis2023')


# sd represents the sludge disposal volume in t/d;
# r1,r2 and r3 represent the proportions of route R1, R2 and R3 respectively
def TAC(sd, r, C0, D0):

    r1, r2 = r[:, 0], r[:, 1]
    r3 = 1 - r1 - r2
    C1, C2, C3 = C0  # The fixed investment costs of routes R1, R2, R3 in the previous year
    D1, D2, D3 = D0  # The processing capacity t/d of routes R1, R2, R3 in the previous year
    # The processing volume t/d of R1, R2, R3 in a future year under the specified scenario
    F1, F2, F3 = sd * r1, sd * r2, sd * r3

    # Calculate the operating costs of routes R1, R2, and R3 based on energy consumption conversion
    cop = (159.9608 * F1 + 26.28 * F2 + 149.9448 * F3) * 365 * 0.5135 / 1e6  # million CNY/y
    cap_r1 = np.copy(C1)
    cap_r2 = np.copy(C2)
    cap_r3 = np.copy(C3)

    # Calculate the investment cost of Route 1
    c1_r1 = F1 > D1
    c2_r1 = (D1==0)
    a_r1 = c1_r1 & c2_r1
    b_r1 = c1_r1 & ~c2_r1
    cap_r1[a_r1] = 55 * F1[a_r1] / 100  # million CNY/y
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
    cap_r2[a_r2] = 40 * F2[a_r2] / 100  # million CNY/y
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
    cap_r3[a_r3] = 40 * F3[a_r3] / 100  # million CNY/y
    cap_r3[b_r3] = C3[b_r3] * (F3[b_r3] / D3[b_r3]) ** 0.6
    cap_r3[~c1_r3] = 0
    cap_rr3 = np.copy(cap_r3)
    cap_rr3[~c1_r3] = C3[~c1_r3]
    F3[~c1_r3] = D3[~c1_r3]

    tac = cop + cap_r1 + cap_r2 + cap_r3
    cap = np.vstack([cap_rr1, cap_rr2, cap_rr3])
    F = np.vstack([F1, F2, F3])

    return tac * 0.14191/1000, cap, F   # total annual costs, billion US$


def Net_Carbon(sd, r):

    r1, r2 = r[:, 0], r[:, 1]
    r3 = 1 - r1 - r2
    co2 = sd * (138.9777522 * r1 + 204.6201959 * r2 + 524.779 * r3) * 365 * 0.2 / 1e6
    return co2/1000  # Mt


def Carbon_energy(sd, r):

    r1, r2 = r[:, 0], r[:, 1]
    r3 = 1 - r1 - r2
    # GHG emissions, Mt/year
    co2 = np.sum(sd * (978.7618 * r1 + 849.248 * r2 + 2101.269 * r3) * 365 * 0.2 / 1e9)
    # energy recovery, TWh/year
    energy = np.sum(sd * (743.391625 * r1 + 28.6 * r2 + 2466.66666666667 * r3) * 365 * 0.2 / 1e9)
    return co2, energy


def Output(xx):
    C = np.zeros((3, 31))
    D = np.zeros((3, 31))
    for i in range(3):
        C[i] = ds[i + 1].to_numpy()
        D[i] = ds[i + 5].to_numpy()

    dd = np.zeros((37, 31))
    for i in range(37):
        dd[i] = df[2024 + i].to_numpy()

    tac = np.zeros((37, 31))
    NGE = np.zeros((37, 31))
    co2 = np.zeros(37)
    energy = np.zeros(37)
    for p in range(37):
        tac[p, :], C, D = TAC(dd[p, :], xx, C, D)
        NGE[p, :] = Net_Carbon(dd[p, :], xx)
        co2[p], energy[p] = Carbon_energy(dd[p, :], xx)
    return np.sum(tac), np.sum(NGE), np.sum(co2) / np.sum(energy)


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=62,
                         n_obj=3,
                         n_constr=31,
                         xl=np.zeros(62),
                         xu=np.ones(62),
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        x_new = x.reshape(-1, 2)
        Res = Output(x_new)
        out["G"] = [np.sum(x_new, axis=1) - 1]
        out["F"] = [Res[0], Res[1], Res[2]]


problem = MyProblem()
algorithm = NSGA3(pop_size=100,
                  n_offsprings=40,
                  ref_dirs=get_reference_directions("das-dennis", 3, n_partitions=12),
                  sampling=FloatRandomSampling(),
                  crossover=SBX(prob=0.8, eta=15),
                  mutation=PolynomialMutation(eta=20),
                  eliminate_duplicates=True
                  )

termination = get_termination("n_gen", 2000)

res = minimize(problem,
               algorithm,
               termination,
               seed=5,
               save_history=True,
               verbose=True
               )

X = res.X
F = res.F
# Visualization
Scatter().add(F).show()
# Output the three objective function values of the Pareto frontier solution to excel
pd.DataFrame(F).to_excel('Pareto_F.xlsx')
# Output the independent variables corresponding to the Pareto frontier solution to excel
pd.DataFrame(X).to_excel('Pareto_X.xlsx')