import pandas as pd
from typing import Union, List, Dict, Any

def dataframe_to_dict(df: pd.DataFrame, keys: Union[str, List[str]], value: str) -> Dict[Any, Any]:
    if isinstance(keys, list):  # If `keys` is a list of columns
        # Use tuples as dictionary keys
        keys_as_tuples = df[keys].apply(tuple, axis=1)
        return dict(zip(keys_as_tuples, df[value]))
    else:  # If `keys` is a single column
        # Use a single column as dictionary keys
        return dict(zip(df[keys], df[value]))



def ramping_rates(generation_data: pd.DataFrame, default: float) -> Dict[Any, float]:
    """Ramp rate per generator, keyed (Country, Technology): the fraction of one unit's
    capacity it can change from one hour to the next.

    Read from the `ramping_rate` column of the generation data. Data without that column
    gives every generator the scalar `default` (ramping_value in inputs/scalars.toml).
    A missing entry stays NaN, so a generator without a rate only fails when ramping is
    actually used (see missing_ramping_rates).
    """
    keys = ["Country", "Technology"]
    if "ramping_rate" not in generation_data.columns:
        return {k: float(default) for k in dataframe_to_dict(generation_data, keys, "UnitCap_MW")}
    return {k: float(v) for k, v in dataframe_to_dict(generation_data, keys, "ramping_rate").items()}


def missing_ramping_rates(rates, generators) -> List[Any]:
    """Generators among `generators` with no finite ramp rate in `rates` (a dict or a scalar)."""
    if not isinstance(rates, dict):
        return [] if pd.notna(rates) else list(generators)
    return [g for g in generators if pd.isna(rates.get(g, float("nan")))]
