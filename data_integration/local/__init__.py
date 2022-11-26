from data_integration import match_table
from utils import concat_dfs
import pandas as pd


def local_attribute_table(sources: dict):
  lat = [(str(s), str(c), str(s) + '_' + str(c)) for s in sources.keys()
         for c in sources[s].columns]
  return pd.DataFrame(lat, columns=['SOURCE', 'LAT', 'SLAT'])


def local_match_table(
    dfs: dict, sim_fns: list, combine: str, threshold=0.5,
    top_k: tuple = None, global_match: str = None, self_match=False,
    src_name='SOURCE', lat_name='LAT', slat_name='SLAT',
    suffixes=('_x', '_y'), sim_name='sim'):
  def lower(a, b, equal=False):
    if equal:
      return a <= b
    else:
      return a < b

  i = 0
  gmt = []
  for kx, vx in dfs.items():
    for ky, vy in dfs.items():
      # if kx < ky:
      if lower(kx, ky, self_match):
        table = match_table(
          vx, vy, sim_fns, combine, threshold, top_k, (global_match, True),
          src_name, lat_name, slat_name, suffixes,
          sim_name=sim_name, src_vals=(kx, ky))
        if i > 0:
          gmt, _ = concat_dfs([gmt, table])
        else:
          gmt = table
        i = i + 1
  return gmt
