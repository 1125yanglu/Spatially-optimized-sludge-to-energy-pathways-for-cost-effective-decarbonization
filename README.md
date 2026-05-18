# Spatially-optimized-sludge-to-energy-pathways-for-cost-effective-decarbonization
Python codes for URPOP, PCGDP and SWD predictions based on quadratic exponential smoothing method and SLD predictions using ARIMAX method are available in the “Forecast 2060” folder.
Python codes for scenario-based Monte Carlo uncertainty simulations of GHG emissions, energy recovery, implementation costs, carbon-energy coefficients and net GHG emissions are available in the “Uncertainty Simulations” folder.
Python codes for optimizing with NSGA-Ⅲ algorithm and evaluating Pareto solutions using the TOPSIS method are available in the “Multi-objective Optimization” folder.

## “Forecast 2060” folder
The Expsmooth.py file uses the second exponential smoothing method to predict urban population (URPOP), per capita gross domestic product (PCGDP), and the volume of sewage treated (SWD). In this file, the original data of URPOP, PCGDP and SWD (URPOP.xlsx, PCGDP.xlsx, SWD.xlsx) need to be called for training and prediction, respectively.
The ARIMAX.py file uses an autoregressive integrated moving average method with exogenous variable to predict provincial sludge disposal (SLD) trajectories from 2014 to 2023. In this file, the original data in the SLD.xlsx file is required to be called for training and prediction. Notably, the SLD.xlsx file includes two sheets. Sheet 1 presents the time series training data for URPOP, PCGDP, SWD, and SLD of a single province from 2005 to 2023, while Sheet 2 provides the predicted data for URPOP, PCGDP, and SWD from 2024 to 2060.
For detailed methods regarding the prediction of URPOP, PCGDP, SWD, and SLD, please refer to the Method section of this manuscript.

## “Uncertainty Simulations” folder
The ScenarioUA_GHG.py file, the ScenarioUA_Power.py file, the ScenarioUA_Cost.py file, the ScenarioUA_CarbonEnergy.py file and the ScenarioUA_NGE.py file are respectively used to perform uncertainty analysis on GHG emissions, energy recovery, implementation costs, carbon-energy coefficients and net GHG emissions over the cumulative period from 2024 to 2060 for the nine scenarios: BUS, ZLS1-4, and OPS1-4. The Data.xlsx file contains the necessary Data for running the above-mentioned python files. For specific usage, you can refer to the comments of each python file.
For a detailed introduction to the scenario-based Monte Carlo uncertainty simulations, please refer to the Method section of this manuscript.

## “Multi-objective Optimization” folder
The MOO.py file is used to perform multi-objective optimization (NSGA-Ⅲ algorithm) of net GHG emissions, implementation costs and carbon-energy coefficients over the cumulative period from 2024 to 2060. The TOPSIS.py file is used to evaluate the Pareto solutions obtained by the MOO.py file. The Data.xlsx file contains the necessary Data for running the above-mentioned python files. For specific usage, you can refer to the comments of each python file.
For a detailed introduction and modeling of NSGA-Ⅲ optimization and TOPSIS method, please refer to the Method section of this manuscript.
