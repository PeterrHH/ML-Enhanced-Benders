import pickle

from gep_config_parser import *
from data_wrangling import dataframe_to_dict

from gep_problem import GEPProblemSet
from gep_problem_operational import GEPOperationalProblemSet
CONFIG_FILE_NAME        = "config.toml"
VISUALIZATION_FILE_NAME = "visualization.toml"


SCALE_FACTORS = {
    "pDemand": 1/1000,  # MW -> GW
    "pGenAva": 1,       # Don't scale
    "pVOLL": 1000,         # kEUR/MWh -> mEUR/GWh
    "pWeight": 1,       # Don't scale
    "pRamping": 1,      # Don't scale
    "pInvCost": 1000,      # kEUR/MW -> mEUR/GW
    "pVarCost": 1000,      # kEUR/MWh -> mEUR/GWh
    "pUnitCap": 1/1000, # MW -> GW
    "pExpCap": 1/1000,  # MW -> GW
    "pImpCap": 1/1000,  # MW -> GW
}

def scale_dict(data_dict, scale_factor):
    return {key: value * scale_factor for key, value in data_dict.items()}


def create_gep_ed_dataset(args, problem_args, inputs, problem_type, save_path, pUnitInvestment_Input = None):
    print("Wrangling the input data")

    # Extract sets
    T = inputs["times"] # [1, 2, 3, ... 8760] ---> 8760
    N = problem_args["N"]
    G = problem_args["G"]
    L = problem_args["L"]

    if not (N or G or L):
        G = inputs["generators"] # [('Country1', 'EnergySource1'), ...] ---> 107
        L = inputs["transmission_lines"] # [('Country1', 'Country2'), ...] ---> 44
        N = inputs["nodes"] # ['Country1', 'Country2', ...] ---> 20
    else:
        # Convert to tuples
        # N = [tuple(pair) for pair in N]
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

    # WOP
    # Scale inversely proportional to times (T)
    pWeight = inputs["representative_period_weight"] / (problem_args["sample_duration"] / 8760)

    pRamping = inputs["ramping_value"]

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

    # if args["scale_problem"]: #! For scaling the problem inputs.
    #     pDemand = scale_dict(pDemand, SCALE_FACTORS["pDemand"])
    #     pGenAva = scale_dict(pGenAva, SCALE_FACTORS["pGenAva"])
    #     pVOLL *= SCALE_FACTORS["pVOLL"]
    #     pWeight *= SCALE_FACTORS["pWeight"]
    #     pRamping *= SCALE_FACTORS["pRamping"]
    #     pInvCost = scale_dict(pInvCost, SCALE_FACTORS["pInvCost"])
    #     pVarCost = scale_dict(pVarCost, SCALE_FACTORS["pVarCost"])
    #     pUnitCap = scale_dict(pUnitCap, SCALE_FACTORS["pUnitCap"])
    #     pExpCap = scale_dict(pExpCap, SCALE_FACTORS["pExpCap"])
    #     pImpCap = scale_dict(pImpCap, SCALE_FACTORS["pImpCap"])


    # We need to sort the dictionaries for changing to tensors!
    pDemand = dict(sorted(pDemand.items()))
    pGenAva = dict(sorted(pGenAva.items()))
    pInvCost = dict(sorted(pInvCost.items()))
    pVarCost = dict(sorted(pVarCost.items()))
    pUnitCap = dict(sorted(pUnitCap.items()))
    pExpCap = dict(sorted(pExpCap.items()))
    pImpCap = dict(sorted(pImpCap.items()))

    print("Creating problem instance")
    if problem_type == "ED":
        data = GEPOperationalProblemSet(args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, pUnitInvestment_Input = pUnitInvestment_Input)
    elif problem_type == "GEP":
        data = GEPProblemSet(problem_args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap)
    
    with open(save_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {save_path}")
    return data

if __name__ == "__main__":
    import json

    ARGS_FILE_NAME = "config.json"
    CONFIG_FILE_NAME = "config.toml"
    with open(ARGS_FILE_NAME, "r") as file:
        args = json.load(file)
    input_data = parse_config(CONFIG_FILE_NAME) # Reads the input data using config.toml's experiment.inputs.data path.
    
    gep_ed_data = input_data["experiment"]["experiments"][0] 
    ED_args = args["ED_args"]
    ED_args["gen_data_constraint"] = True
    print(f"ED args gen with node cosntraint: {ED_args['gen_data_node_constraint']}")
    data_save_path = (f"data/ED_data/Constraint/3Loc/Test.pkl")
    data = create_gep_ed_dataset(args=args, problem_args=ED_args, inputs=gep_ed_data, problem_type=args["problem_type"], save_path=data_save_path, pUnitInvestment_Input = None)