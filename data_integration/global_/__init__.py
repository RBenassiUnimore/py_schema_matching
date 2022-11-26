import pandas as pd
from data_integration import match_table
from utils import concat_dfs


def global_match_table(
    local_dfs: dict, global_df: pd.DataFrame,
    sim_fns: list, combine: str, threshold=0.5,
    top_k: tuple = None, global_match: str = None,
    gat_name='GAT', src_name='SOURCE', lat_name='LAT',
    slat_name='SLAT', sim_name='sim'):
  gm = None if global_match is None else (global_match, False)
  i = 0
  gmt = []
  for k, v in local_dfs.items():
    table = match_table(
      v, global_df, sim_fns, combine, threshold, top_k, gm,
      src_names=(src_name, 'SG'), lat_names=(lat_name, gat_name),
      slat_names=(slat_name, 'SLG'), sim_name=sim_name, src_vals=(k, 'G'))
    if i > 0:
      gmt, _ = concat_dfs([gmt, table])
    else:
      gmt = table
    i = i + 1

  return gmt[[gat_name, src_name, lat_name, slat_name, sim_name]]
