import time
import numpy as np
from gep_config_parser import *
from data_wrangling import dataframe_to_dict, ramping_rates, missing_ramping_rates
import pyomo.environ as pyo
from pyomo.environ import Var, Constraint
from pyomo.repn import generate_standard_repn
import json

DEVICE = 'cpu'

CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"

HIGHS  = "HiGHS"
GUROBI = "Gurobi"

SCALE_FACTORS = {
    "pDemand": 1/1000,  # MW -> GW
    "pGenAva": 1,       # Don't scale
    "pVOLL": 1/1000,         # kEUR/MWh -> mEUR/GWh
    "pWeight": 1,       # Don't scale
    "pRamping": 1,      # Don't scale
    "pInvCost": 1/1000,      # kEUR/MW -> mEUR/GW
    "pVarCost": 1/1000,      # kEUR/MWh -> mEUR/GWh
    "pUnitCap": 1/1000, # MW -> GW
    "pExpCap": 1/1000,  # MW -> GW
    "pImpCap": 1/1000,  # MW -> GW
}

def scale_dict(data_dict, scale_factor):
    return {key: value * scale_factor for key, value in data_dict.items()}

## Step 0: Activate environment - ensure consistency accross computers
# print("Reading the data")
# print("Activating the environment")

## Step 1: parse the input data
print("Parsing the config file")

data = parse_config(CONFIG_FILE_NAME)
experiment = data["experiment"]
outputs_config = data["outputs_config"]

print("Initializing the solver")
optimizer_name = data["optimizer_config"]["solver"]

# Determine the optimizer
if optimizer_name == HIGHS:
    raise NotImplementedError(f"{optimizer_name}: Not implemented")
elif optimizer_name == GUROBI:
    
    print(f"Using {GUROBI}")
    optimizer = "gurobi_direct"
else:
    raise ValueError(f"{optimizer_name}: Not implemented")

def prep_data(args, inputs, N=None, G=None, L=None, scale=True):
        print("Wrangling the input data")

        # Extract sets
        T = inputs["times"] # [1, 2, 3, ... 8761] ---> 8761

        if not (N or G or L):
            G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
            L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
            N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20
        else:
            # Convert to tuples
            G = [tuple(pair) for pair in G]
            L = [tuple(pair) for pair in L]


        # Extract time series data
        pDemand = dataframe_to_dict(
            inputs["demand_data"],
            keys=["Country", "Time"],
            value="Demand_MW"
        )
        pGenAva = dataframe_to_dict(
            inputs["generation_availability_data"],
            keys=["Country", "Technology", "Time"],
            value="Availability_pu"
        )

        # Extract scalar parameters
        pVOLL = inputs["value_of_lost_load"]
        pWeight = inputs["representative_period_weight"] / (args["sample_duration"] / len(T))
        #! Per generator, from the ramping_rate column of the generation data (scalar fallback).
        pRamping = dict(sorted(ramping_rates(inputs["generation_data"], inputs["ramping_value"]).items()))
        if inputs["ramping"] == "true" and missing_ramping_rates(pRamping, G):
            raise ValueError(f"No ramping_rate for {missing_ramping_rates(pRamping, G)} in the generation data.")

        # Extract generator parameters
        pInvCost = dataframe_to_dict(
            inputs["generation_data"],
            keys=["Country", "Technology"],
            value="InvCost_kEUR_MW_year"
        )
        pVarCost = dataframe_to_dict(
            inputs["generation_data"],
            keys=["Country", "Technology"],
            value="VarCost_kEUR_per_MWh"
        )
        pUnitCap = dataframe_to_dict(
            inputs["generation_data"],
            keys=["Country", "Technology"],
            value="UnitCap_MW"
        )

        # Extract line parameters
        pExpCap = dataframe_to_dict(
            inputs["transmission_lines_data"],
            keys=["CountryA", "CountryB"],
            value="ExpCap_MW"
        )
        pImpCap = dataframe_to_dict(
            inputs["transmission_lines_data"],
            keys=["CountryA", "CountryB"],
            value="ImpCap_MW"
        )

        # We need to sort the dictionaries for changing to tensors!
        pDemand = dict(sorted(pDemand.items()))
        pGenAva = dict(sorted(pGenAva.items()))
        pInvCost = dict(sorted(pInvCost.items()))
        pVarCost = dict(sorted(pVarCost.items()))
        pUnitCap = dict(sorted(pUnitCap.items()))
        pExpCap = dict(sorted(pExpCap.items()))
        pImpCap = dict(sorted(pImpCap.items()))

        if scale:
            pDemand = scale_dict(pDemand, SCALE_FACTORS["pDemand"])
            pGenAva = scale_dict(pGenAva, SCALE_FACTORS["pGenAva"])
            pVOLL *= SCALE_FACTORS["pVOLL"]
            pWeight *= SCALE_FACTORS["pWeight"]
            pRamping = scale_dict(pRamping, SCALE_FACTORS["pRamping"])
            pInvCost = scale_dict(pInvCost, SCALE_FACTORS["pInvCost"])
            pVarCost = scale_dict(pVarCost, SCALE_FACTORS["pVarCost"])
            pUnitCap = scale_dict(pUnitCap, SCALE_FACTORS["pUnitCap"])
            pExpCap = scale_dict(pExpCap, SCALE_FACTORS["pExpCap"])
            pImpCap = scale_dict(pImpCap, SCALE_FACTORS["pImpCap"])


        return T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap

def run_model(inputs, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, constant_gen_inv=False):

    # ! Keep GenInv constant, instead of a decision variable (change to operational problem):
    if constant_gen_inv:
        pGenInv = {('BEL', 'SunPV'): 4130.05009001755, ('GER', 'SunPV'): 11232.550865341998}
        # With Gas and Sun:
    
    # Extract optimizer attributes
    attributes = data["optimizer_config"][optimizer_name]

    # Add the log file attribute
    attributes["LogFile"] = inputs["output_log"]

    # Check the crossover setting
    if inputs["crossover"] != "gurobi":
        attributes["Crossover"] = 0
        attributes["FeasibilityTol"] = 1e-9

    # Initialize the model with the optimizer and attributes
    model, solver = initialize_model(optimizer, attributes)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)    # Add dual problem.
    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)    # Add reduced costs.

    print("Populating the model")

    # Create variables
    print("Adding model variables")

    # Investment cost variable (non-negative)
    model.vInvCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # Operational cost variable (non-negative)
    model.vOpeCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # Generator investment variables
    if not constant_gen_inv:
        if inputs["relaxed"] == "true":
            model.vGenInv = pyo.Var(G, within=pyo.NonNegativeReals, initialize=0)
        else:
            model.vGenInv = pyo.Var(G, within=pyo.Integers, bounds=(0, None), initialize=0)

    # Generator production variables (non-negative)
    model.vGenProd = pyo.Var(G, T, within=pyo.NonNegativeReals, initialize=0)

    # Ensure lineflow is within max import and export capacities
    # 3.1d & 3.1e
    # Since L has a tuple of two countries as index, pyomo unpacks it
    def lineFlowBounds(model, l0, l1, t):
        return (-pImpCap[(l0, l1)], pExpCap[(l0, l1)])

    # Transmission line flow variables with bounds    
    model.vLineFlow = pyo.Var(L, T, bounds=lineFlowBounds)

    # Loss of load variables with bounds based on demand
    model.vLossLoad = pyo.Var(
        N, T, 
        initialize=0, 
        bounds=lambda model, n, t: (0, pDemand[(n, t)])
    )

    # Formulate objective
    print("Formulating the objective")
    model.obj = pyo.Objective(expr=model.vInvCost + model.vOpeCost, sense=pyo.minimize)

    # Constraints
    print("Adding model constraints")

    # Investment costs
    # Sum_{g in G} IC_g * UCAP_g * ui_g
    # Investment cost (€/MW) * capacity of a unit (MW) * nr of units = Investment cost at g (€)
    if constant_gen_inv:
        def eInvCost_rule(model):
            return model.vInvCost == sum(
                pInvCost[g] * pUnitCap[g] * pGenInv[g] for g in G
            )
    else:
        def eInvCost_rule(model):
            return model.vInvCost == sum(
                pInvCost[g] * pUnitCap[g] * model.vGenInv[g] for g in G
            )
    model.eInvCost = pyo.Constraint(rule=eInvCost_rule)

    # Operating costs
    # WOP * (Sum_{g in G, t in T} (PC_g * p_{g, t}) + Sum_{n in N, t in T} (MDC * md_{n,t}))
    # Period weight * (variable production cost (€/MWh) * energy generation (MW) + 
    #                  cost of missed demand (€/MW) * missed demand (€/MW))
    def eOpeCost_rule(model):
        return model.vOpeCost == pWeight * (
            sum(pVarCost[g] * model.vGenProd[g, t] for g in G for t in T)
            + sum(pVOLL * model.vLossLoad[n, t] for n in N for t in T)
        )
    model.eOpeCost = pyo.Constraint(rule=eOpeCost_rule)

    # (3.1c)
    # Ensure energy balance at each node
    # Demand = generation + transmissionB->A - transmissionA-B + missed_demand
    def eNodeBal_rule(model, n, t):
        return (
            sum(model.vGenProd[g, t] for g in G if g[0] == n)
            + sum(model.vLineFlow[l, t] for l in L if l[1] == n)
            - sum(model.vLineFlow[l, t] for l in L if l[0] == n)
            + model.vLossLoad[n, t]
            == pDemand[(n, t)]
        )
    model.eNodeBal = pyo.Constraint(N, T, rule=eNodeBal_rule)

    # (3.1b)
    # Ensure production never exceeds capacity
    if constant_gen_inv:
        def eMaxProd_rule(model, g0, g1, t):
            availability = pGenAva.get((g0, g1, t), 1.0)  # Default availability to 1.0
            return model.vGenProd[(g0, g1), t] <= availability * pUnitCap[(g0, g1)] * pGenInv[(g0, g1)]
    else:
        def eMaxProd_rule(model, g0, g1, t):
            availability = pGenAva.get((g0, g1, t), 1.0)  # Default availability to 1.0
            return model.vGenProd[(g0, g1), t] <= availability * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
        
    model.eMaxProd = pyo.Constraint(G, T, rule=eMaxProd_rule)

    # Ramping constraints
    # No large changes in production between timesteps
    if constant_gen_inv:
        if inputs["ramping"] == "true":
        # Ramping up (3.1g)
            def eRampingUp_rule(model, g0, g1, t):
                if t == T[0]:  # Skip the first time step for ramping constraints
                    return pyo.Constraint.Skip
                return (
                    model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                    <= pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * pGenInv[(g0, g1)]
                )
            model.eRampingUp = pyo.Constraint(G, T, rule=eRampingUp_rule)

            # Ramping down (3.1f)
            def eRampingDown_rule(model, g0, g1, t):
                if t == T[0]:  # Skip the first time step for ramping constraints
                    return pyo.Constraint.Skip
                return (
                    -pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * pGenInv[(g0, g1)]
                    <= model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                )
            model.eRampingDown = pyo.Constraint(G, T, rule=eRampingDown_rule)
    else:
        if inputs["ramping"] == "true":
            # Ramping up (3.1g)
            def eRampingUp_rule(model, g0, g1, t):
                if t == T[0]:  # Skip the first time step for ramping constraints
                    return pyo.Constraint.Skip
                return (
                    model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                    <= pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
                )
            model.eRampingUp = pyo.Constraint(G, T, rule=eRampingUp_rule)

            # Ramping down (3.1f)
            def eRampingDown_rule(model, g0, g1, t):
                if t == T[0]:  # Skip the first time step for ramping constraints
                    return pyo.Constraint.Skip
                return (
                    -pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
                    <= model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                )
            model.eRampingDown = pyo.Constraint(G, T, rule=eRampingDown_rule)

    ## Step 4: Solve
    print("Solving the optimization problem")
    results = solver.solve(model, tee=False)
    time_taken = solver._solver_model.Runtime

    print(f"Objective Value: {model.obj()}")
    return model, solver, time_taken

def get_variable_values_as_list(var):
    # Check if the variable is indexed
    if var.is_indexed():
        return [var[idx].value for idx in var]
    else:
        # For scalar variables
        return [var.value]

def run_model_no_bounds(inputs, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap):
    """This model is a slower implementation, which does not use any bounds (domains), but is instead modeled using only constraints, so that it is in line with the GEP problem for PDL, 
    where all of the domains are processed as constraints, and the inequality constraints are rewritten to the form <= 0, and equality constraints to the form = 0.
    We do this so that we can directly compare the solver output to PDL.
    """
    
    # Extract optimizer attributes
    attributes = data["optimizer_config"][optimizer_name]

    # Add the log file attribute
    attributes["LogFile"] = inputs["output_log"]

    # Check the crossover setting
    if inputs["crossover"] != "gurobi":
        attributes["Crossover"] = 1
        attributes["FeasibilityTol"] = 1e-9

    # Initialize the model with the optimizer and attributes
    model, solver = initialize_model(optimizer, attributes)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)    # Add dual problem.
    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)    # Add reduced costs.

    print("Populating the model")

    # Create variables
    print("Adding model variables")

    # Investment cost variable (non-negative)
    model.vInvCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # Operational cost variable (non-negative)
    model.vOpeCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # if not constant_gen_inv:
    # Generator investment variables
    if inputs["relaxed"] == "true":
        # model.vGenInv = pyo.Var(G, within=pyo.NonNegativeReals, initialize=0)
        model.vGenInv = pyo.Var(G, initialize=0) #! Remove the domain.
    else:
        model.vGenInv = pyo.Var(G, within=pyo.Integers, bounds=(0, None), initialize=0)

    # Generator production variables (non-negative)
    # model.vGenProd = pyo.Var(G, T, within=pyo.NonNegativeReals, initialize=0)
    model.vGenProd = pyo.Var(G, T, initialize=0) #! Remove the domain

    # Ensure lineflow is within max import and export capacities
    # 3.1d & 3.1e
    # Since L has a tuple of two countries as index, pyomo unpacks it
    def lineFlowBounds(model, l0, l1, t):
        return (-pImpCap[(l0, l1)], pExpCap[(l0, l1)])

    # Transmission line flow variables with bounds    
    # model.vLineFlow = pyo.Var(L, T, bounds=lineFlowBounds)
    model.vLineFlow = pyo.Var(L, T) #! Remove the bounds

    # Loss of load variables with bounds based on demand
    model.vLossLoad = pyo.Var(
        N, T, 
        initialize=0, 
        # bounds=lambda model, n, t: (0, pDemand[(n, t)]) #! Remove the bounds
    )

    # Formulate objective
    print("Formulating the objective")
    # model.obj = pyo.Objective(expr=model.vInvCost + model.vOpeCost, sense=pyo.minimize)
    model.obj = pyo.Objective(expr=(sum(pInvCost[g] * pUnitCap[g] * model.vGenInv[g] for g in G) + 
                                    pWeight * (sum(pVarCost[g] * model.vGenProd[g, t] for g in G for t in T) + sum(pVOLL * model.vLossLoad[n, t] for n in N for t in T))
                                    ),
                                    sense=pyo.minimize
                                    )

    # Constraints
    print("Adding model constraints")

    # Investment costs
    # Sum_{g in G} IC_g * UCAP_g * ui_g
    # Investment cost (€/MW) * capacity of a unit (MW) * nr of units = Investment cost at g (€)
    # def eInvCost_rule(model):
    #     return model.vInvCost == sum(
    #         pInvCost[g] * pUnitCap[g] * model.vGenInv[g] for g in G
    #     )
    # model.eInvCost = pyo.Constraint(rule=eInvCost_rule)

    # # Operating costs
    # # WOP * (Sum_{g in G, t in T} (PC_g * p_{g, t}) + Sum_{n in N, t in T} (MDC * md_{n,t}))
    # # Period weight * (variable production cost (€/MWh) * energy generation (MW) + 
    # #                  cost of missed demand (€/MW) * missed demand (€/MW))
    # def eOpeCost_rule(model):
    #     return model.vOpeCost == pWeight * (
    #         sum(pVarCost[g] * model.vGenProd[g, t] for g in G for t in T)
    #         + sum(pVOLL * model.vLossLoad[n, t] for n in N for t in T)
    #     )
    # model.eOpeCost = pyo.Constraint(rule=eOpeCost_rule)

    # (3.1b)
    # Ensure production never exceeds capacity
    def eMaxProd_rule(model, g0, g1, t):
        availability = pGenAva.get((g0, g1, t), 1.0)  # Default availability to 1.0
        # return model.vGenProd[(g0, g1), t] <= availability * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)] #! Rewrite to <= 0
        return model.vGenProd[(g0, g1), t] - (availability * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]) <= 0
        
    model.eMaxProd = pyo.Constraint(G, T, rule=eMaxProd_rule)


    # (3.1c)
    # Ensure energy balance at each node
    # Demand = generation + transmissionB->A - transmissionA-B + missed_demand
    def eNodeBal_rule(model, n, t):
        return (
            # sum(model.vGenProd[g, t] for g in G if g[0] == n)
            # + sum(model.vLineFlow[l, t] for l in L if l[1] == n)
            # - sum(model.vLineFlow[l, t] for l in L if l[0] == n)
            # + model.vLossLoad[n, t]
            # == pDemand[(n, t)] #! Rewrite to form  = 0
            sum(model.vGenProd[g, t] for g in G if g[0] == n)
            + sum(model.vLineFlow[l, t] for l in L if l[1] == n)
            - sum(model.vLineFlow[l, t] for l in L if l[0] == n)
            + model.vLossLoad[n, t]
            - pDemand[(n, t)]
            == 0
        )
    model.eNodeBal = pyo.Constraint(N, T, rule=eNodeBal_rule)

    # (3.1d)
    # Ensure lineflow lower bound
    def eLineFlowLB_rule(model, l0, l1, t):
        return -1 * pImpCap[(l0, l1)] - model.vLineFlow[(l0, l1), t] <= 0

    model.eLineFlowLB = pyo.Constraint(L, T, rule=eLineFlowLB_rule)

    # (3.1e)
    # Ensure lineflow upper bound
    def eLineFlowUB_rule(model, l0, l1, t):
        return model.vLineFlow[(l0, l1), t] - pExpCap[(l0, l1)] <= 0

    model.eLineFlowUB = pyo.Constraint(L, T, rule=eLineFlowUB_rule)

    # Ramping constraints
    # No large changes in production between timesteps
    if inputs["ramping"] == "true":
        # Ramping up (3.1g)
        def eRampingUp_rule(model, g0, g1, t):
            if t == T[0]:  # Skip the first time step for ramping constraints
                return pyo.Constraint.Skip
            return (
                # model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                # <= pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)] # ! Transform to <= 0
                (model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]) - (pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]) <= 0
            )
        model.eRampingUp = pyo.Constraint(G, T, rule=eRampingUp_rule)

        # Ramping down (3.1f)
        def eRampingDown_rule(model, g0, g1, t):
            if t == T[0]:  # Skip the first time step for ramping constraints
                return pyo.Constraint.Skip
            return (
                # -pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
                # <= model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]] # ! Transform to <= 0
                (-pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]) - (model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]) <= 0
            )
        model.eRampingDown = pyo.Constraint(G, T, rule=eRampingDown_rule)

    def eGenProdPositive_rule(model, g0, g1, t):
        return -1 * model.vGenProd[(g0, g1), t] <= 0
    
    model.eGenProdPositive = pyo.Constraint(G, T, rule=eGenProdPositive_rule)

    def eMissedDemandPositive_rule(model, n, t):
        return -1 * model.vLossLoad[n, t] <= 0
    model.eMissedDemandPositive = pyo.Constraint(N, T, rule=eMissedDemandPositive_rule)

    def eMissedDemandLeqDemand_rule(model, n, t):
        return model.vLossLoad[n, t] - pDemand[n, t] <= 0
    model.eMissedDemandLeqDemand = pyo.Constraint(N, T, rule=eMissedDemandLeqDemand_rule)

    # if not constant_gen_inv: 
    def eGenInvPositive_rule(model, g0, g1):
        return -1 * model.vGenInv[(g0, g1)] <= 0

    model.eGenInvPositive = pyo.Constraint(G, rule=eGenInvPositive_rule)

    ## Step 4: Solve
    print("Solving the optimization problem")
    # results = solver.solve(model, tee=True)
    results = solver.solve(model, tee=False)
    time_taken = solver._solver_model.Runtime

    print(f"Objective Value: {model.obj()}")
    return model, solver, time_taken

def run_operational_model_no_bounds(inputs, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, pGenInv):

    # ! Keep GenInv constant, instead of a decision variable (change to operational problem):
    # pGenInv = {('BEL', 'SunPV'): 4130.05009001755, ('GER', 'SunPV'): 11232.550865341998}
    # With gas and sun:
    # 361.86402118882216, 0.0, 26.82598871780973, 164.8367340122868
    # pGenInv = {('BEL', 'SunPV'): 361.86402118882216, ('GER', 'SunPV'): 0.0, ('BEL', 'Gas'): 26.82598871780973, ('GER', 'Gas'): 164.8367340122868}
    
    # Extract optimizer attributes
    attributes = data["optimizer_config"][optimizer_name]

    # Add the log file attribute
    attributes["LogFile"] = inputs["output_log"]

    # Check the crossover setting
    # if inputs["crossover"] != "gurobi":
        # attributes["Crossover"] = 1
    attributes['Method'] = 1  # Dual Simplex
    attributes["FeasibilityTol"] = 1e-09
    

    # Initialize the model with the optimizer and attributes
    model, solver = initialize_model(optimizer, attributes)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)    # Add dual problem.
    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)    # Add reduced costs.

    print("Populating the model")

    # Create variables
    print("Adding model variables")

    # Investment cost variable (non-negative)
    # model.vInvCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # Operational cost variable (non-negative)
    model.vOpeCost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)

    # if not constant_gen_inv:
    # Generator investment variables
    if inputs["relaxed"] == "true":
        # model.vGenInv = pyo.Var(G, within=pyo.NonNegativeReals, initialize=0)
        model.vGenInv = pyo.Var(G, initialize=0) #! Remove the domain.
    else:
        model.vGenInv = pyo.Var(G, within=pyo.Integers, bounds=(0, None), initialize=0)

    # Generator production variables (non-negative)
    # model.vGenProd = pyo.Var(G, T, within=pyo.NonNegativeReals, initialize=0)
    model.vGenProd = pyo.Var(G, T, initialize=0) #! Remove the domain

    # Transmission line flow variables with bounds    
    # model.vLineFlow = pyo.Var(L, T, bounds=lineFlowBounds)
    model.vLineFlow = pyo.Var(L, T) #! Remove the bounds

    # Loss of load variables with bounds based on demand
    model.vLossLoad = pyo.Var(
        N, T, 
        initialize=0, 
        # bounds=lambda model, n, t: (0, pDemand[(n, t)]) #! Remove the bounds
    )

    # Formulate objective
    print("Formulating the objective")
    model.obj = pyo.Objective(expr=model.vOpeCost, sense=pyo.minimize)

    # Constraints
    print("Adding model constraints")

    # Investment costs
    # Sum_{g in G} IC_g * UCAP_g * ui_g
    # Investment cost (€/MW) * capacity of a unit (MW) * nr of units = Investment cost at g (€)
    # if constant_gen_inv:
    #     def eInvCost_rule(model):
    #         return model.vInvCost == sum(
    #             pInvCost[g] * pUnitCap[g] * pGenInv[g] for g in G
    #         )
    # else:
    # def eInvCost_rule(model):
        # return model.vInvCost == sum(
            # pInvCost[g] * pUnitCap[g] * model.vGenInv[g] for g in G
        # )
    # model.eInvCost = pyo.Constraint(rule=eInvCost_rule)

    # Operating costs
    # WOP * (Sum_{g in G, t in T} (PC_g * p_{g, t}) + Sum_{n in N, t in T} (MDC * md_{n,t}))
    # Period weight * (variable production cost (€/MWh) * energy generation (MW) + 
    #                  cost of missed demand (€/MW) * missed demand (€/MW))
    def eOpeCost_rule(model): #! Should we add pWeight here, or not?
        return model.vOpeCost == (
            sum(pVarCost[g] * model.vGenProd[g, t] for g in G for t in T)
            + sum(pVOLL * model.vLossLoad[n, t] for n in N for t in T)
        )
    model.eOpeCost = pyo.Constraint(rule=eOpeCost_rule)

    # (3.1b)
    # Ensure production never exceeds capacity
    def eMaxProd_rule(model, g0, g1, t):
        availability = pGenAva.get((g0, g1, t), 1.0)  # Default availability to 1.0
        # return model.vGenProd[(g0, g1), t] <= availability * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)] #! Rewrite to <= 0
        return model.vGenProd[(g0, g1), t] - (availability * pUnitCap[(g0, g1)] * pGenInv[G.index((g0, g1))]) <= 0
        
    model.eMaxProd = pyo.Constraint(G, T, rule=eMaxProd_rule)


    # (3.1c)
    # Ensure energy balance at each node
    # Demand = generation + transmissionB->A - transmissionA-B + missed_demand
    def eNodeBal_rule(model, n, t):
        return (
            # sum(model.vGenProd[g, t] for g in G if g[0] == n)
            # + sum(model.vLineFlow[l, t] for l in L if l[1] == n)
            # - sum(model.vLineFlow[l, t] for l in L if l[0] == n)
            # + model.vLossLoad[n, t]
            # == pDemand[(n, t)] #! Rewrite to form  = 0
            sum(model.vGenProd[g, t] for g in G if g[0] == n)
            + sum(model.vLineFlow[l, t] for l in L if l[1] == n)
            - sum(model.vLineFlow[l, t] for l in L if l[0] == n)
            + model.vLossLoad[n, t]
            - pDemand[(n, t)]
            == 0
        )
    model.eNodeBal = pyo.Constraint(N, T, rule=eNodeBal_rule)

    # (3.1d)
    # Ensure lineflow lower bound
    def eLineFlowLB_rule(model, l0, l1, t):
        return -1 * pImpCap[(l0, l1)] - model.vLineFlow[(l0, l1), t] <= 0

    model.eLineFlowLB = pyo.Constraint(L, T, rule=eLineFlowLB_rule)

    # (3.1e)
    # Ensure lineflow upper bound
    def eLineFlowUB_rule(model, l0, l1, t):
        return model.vLineFlow[(l0, l1), t] - pExpCap[(l0, l1)] <= 0

    model.eLineFlowUB = pyo.Constraint(L, T, rule=eLineFlowUB_rule)

    # Ramping constraints
    # No large changes in production between timesteps
    if inputs["ramping"] == "true":
        # Ramping up (3.1g)
        def eRampingUp_rule(model, g0, g1, t):
            if t == T[0]:  # Skip the first time step for ramping constraints
                return pyo.Constraint.Skip
            return (
                # model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]
                # <= pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)] # ! Transform to <= 0
                (model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]) - (pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]) <= 0
            )
        model.eRampingUp = pyo.Constraint(G, T, rule=eRampingUp_rule)

        # Ramping down (3.1f)
        def eRampingDown_rule(model, g0, g1, t):
            if t == T[0]:  # Skip the first time step for ramping constraints
                return pyo.Constraint.Skip
            return (
                # -pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]
                # <= model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]] # ! Transform to <= 0
                (-pRamping[(g0, g1)] * pUnitCap[(g0, g1)] * model.vGenInv[(g0, g1)]) - (model.vGenProd[(g0, g1), t] - model.vGenProd[(g0, g1), T[T.index(t) - 1]]) <= 0
            )
        model.eRampingDown = pyo.Constraint(G, T, rule=eRampingDown_rule)

    def eGenProdPositive_rule(model, g0, g1, t):
        return -1 * model.vGenProd[(g0, g1), t] <= 0
    
    model.eGenProdPositive = pyo.Constraint(G, T, rule=eGenProdPositive_rule)

    def eMissedDemandPositive_rule(model, n, t):
        return -1 * model.vLossLoad[n, t] <= 0
    model.eMissedDemandPositive = pyo.Constraint(N, T, rule=eMissedDemandPositive_rule)

    def eMissedDemandLeqDemand_rule(model, n, t):
        return model.vLossLoad[n, t] - pDemand[n, t] <= 0
    model.eMissedDemandLeqDemand = pyo.Constraint(N, T, rule=eMissedDemandLeqDemand_rule)

    ## Step 4: Solve
    print("Solving the optimization problem")
    # results = solver.solve(model, tee=True)
    results = solver.solve(model, tee=False)
    time_taken = solver._solver_model.Runtime

    print(f"Objective Value: {model.obj()}")
    return model, solver, time_taken

def get_variable_values_as_list(var):
    # Check if the variable is indexed
    if var.is_indexed():
        return [var[idx].value for idx in var]
    else:
        # For scalar variables
        return [var.value]
    
def extract_constraint_matrix_split(model):
    # Gather active variables and constraints.
    var_list = list(model.component_data_objects(Var, active=True))
    cons_list = list(model.component_data_objects(Constraint, active=True))
    
    # Build a dictionary mapping variable id to its index.
    var_index = {id(var): i for i, var in enumerate(var_list)}
    num_vars = len(var_list)
    
    # Lists to collect equality and inequality rows, RHS values, and inequality senses.
    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_rhs = []
    ineq_senses = []
    
    for con in cons_list:
        repn = generate_standard_repn(con.body)
        # Build row vector (size = num_vars) for the constraint.
        row = np.zeros(num_vars)
        for coef, var in zip(repn.linear_coefs, repn.linear_vars):
            j = var_index.get(id(var))
            if j is None:
                raise KeyError("Variable not found in var_list")
            row[j] = coef
        
        # Determine the RHS value and, for inequalities, the sense.
        if con.equality:
            rhs_val = con.upper - repn.constant
            eq_rows.append(row)
            eq_rhs.append(rhs_val)
        elif con.has_ub() and not con.has_lb():
            rhs_val = con.upper - repn.constant
            ineq_rows.append(row)
            ineq_rhs.append(rhs_val)
            ineq_senses.append('<=')
        elif con.has_lb() and not con.has_ub():
            rhs_val = con.lower - repn.constant
            ineq_rows.append(row)
            ineq_rhs.append(rhs_val)
            ineq_senses.append('>=')
        else:
            # For range constraints (both lower and upper defined), here we choose the upper bound,
            # and mark it as 'range'. You could split these into two constraints if needed.
            rhs_val = con.upper - repn.constant
            ineq_rows.append(row)
            ineq_rhs.append(rhs_val)
            ineq_senses.append('range')
    
    # Convert collected rows to matrices/vectors.
    eq_cm = np.array(eq_rows)
    ineq_cm = np.array(ineq_rows)
    eq_rhs = np.array(eq_rhs)
    ineq_rhs = np.array(ineq_rhs)
    
    return eq_cm, ineq_cm, eq_rhs, ineq_rhs, ineq_senses

def extract_obj_coeff(model):
    # Get all active variables in the model
    var_list = list(model.component_data_objects(Var, active=True))
    # Create a dictionary mapping variable id to index.
    var_index = {id(var): i for i, var in enumerate(var_list)}
    num_vars = len(var_list)
    
    # Initialize the objective coefficient vector.
    obj_coeff = np.zeros(num_vars)
    
    # Generate the standard representation of the objective expression.
    repn_obj = generate_standard_repn(model.obj.expr)
    
    # Fill the coefficient vector.
    for coef, var in zip(repn_obj.linear_coefs, repn_obj.linear_vars):
        j = var_index.get(id(var))
        if j is None:
            raise KeyError("Variable not found in var_list for objective.")
        obj_coeff[j] = coef
    
    return obj_coeff

def solve_matrix_problem(eq_cm, ineq_cm, eq_rhs, ineq_rhs, obj_coeff):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    m = gp.Model("Matrix problem")

    # Create variables
    x = m.addMVar(shape=len(obj_coeff), vtype=GRB.CONTINUOUS, name="x")

    # Set objective
    obj = np.array(obj_coeff)
    print(obj)
    m.setObjective(obj @ x, GRB.MINIMIZE)

    # Add ineq constraints
    A = np.array(ineq_cm)
    b = np.array(ineq_rhs)
    m.addConstr(A @ x <= b, name="ineq")

    # print(A)
    # print(b)

    # Add eq constraints
    A = np.array(eq_cm)
    b = np.array(eq_rhs)
    m.addConstr(A @ x == b, name="eq")

    # print(A)
    # print(b)

    # Optimize model
    m.optimize()

    print(x.X)
    print(f"Obj: {m.ObjVal:g}")

    return x.X, m.ObjVal
    
if __name__ == "__main__":
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        with open("config.json", "r") as file:
            args = json.load(file)

        T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap = prep_data(args, experiment_instance, N=args["N"], G=args["G"], L=args["L"], scale=args["scale_problem"])

        T_ranges = [range(i, i + args["sample_duration"], 1) for i in range(1, len(T), args["sample_duration"])]
        for t in T_ranges[:1]:
            # Run one experiment for j repeats
            model, solver, time_taken = run_model_no_bounds(experiment_instance, t, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap)

            eq_cm, ineq_cm, eq_rhs, ineq_rhs, ineq_senses = extract_constraint_matrix_split(model)
            obj_coeff = extract_obj_coeff(model)

            sol, obj = solve_matrix_problem(eq_cm, ineq_cm, eq_rhs, ineq_rhs, obj_coeff)

            print(model.obj())
            print(obj)
    