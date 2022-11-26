import pandas as pd

from py_schema_matching.correspondances.local import thresholding, top_k
from py_schema_matching.correspondances.global_ import simmetric_best_match, stable_marriage
from py_schema_matching.similarity_table.combined import ensemble_sim_table


def to_gmt(match_matrix: pd.DataFrame, gat_name='GAT', src_name='SOURCE',
           lat_name='LAT', slat_name='SLAT'):
  gmt = pd.DataFrame(columns=[gat_name, src_name, lat_name, slat_name])

  for x in match_matrix.index:
    for y in match_matrix.columns:
      for z in match_matrix.loc[x][y]:
        gmt.loc[len(gmt)] = [x, y, z, str(y) + '_' + z]
  return gmt


def to_gmm(match_table: pd.DataFrame, gat_name='GAT', src_name='SOURCE',
           lat_name='LAT', slat_name='SLAT'):
  gmm = match_table.groupby([gat_name, src_name])[lat_name].agg(
    list).unstack(src_name)
  for c in gmm.columns:
    gmm.loc[gmm[c].isnull(), [c]] = gmm.loc[gmm[c].isnull(), c].apply(lambda x: [])
  return gmm


def match_table(
    df0: pd.DataFrame, df1: pd.DataFrame,
    sim_fns: list, combine, threshold=0.5,
    topk: tuple = None, global_match: tuple = None,
    src_prefix='SOURCE', lat_prefix='LAT', slat_prefix='SLAT',
    suffixes=('_x', '_y'),
    src_names: list = None, lat_names: list = None, slat_names: list = None,
    sim_name='sim', src_vals=('Sx', 'Sy')):
  if src_names is None:
    src_name_l = src_prefix + suffixes[0]
    src_name_r = src_prefix + suffixes[1]
  else:
    src_name_l, src_name_r = src_names[0], src_names[1]

  if lat_names is None:
    lat_name_l = lat_prefix + suffixes[0]
    lat_name_r = lat_prefix + suffixes[1]
  else:
    lat_name_l, lat_name_r = lat_names[0], lat_names[1]

  if slat_names is None:
    slat_name_l = slat_prefix + suffixes[0]
    slat_name_r = slat_prefix + suffixes[1]
  else:
    slat_name_l, slat_name_r = slat_names[0], slat_names[1]

  gmt = ensemble_sim_table(df0, df1, sim_fns, combine,
                           lat_name_l, lat_name_r, sim_name)

  gmt[src_name_l] = src_vals[0]
  gmt[src_name_r] = src_vals[1]
  gmt[slat_name_l] = gmt[src_name_l] + '_' + gmt[lat_name_l]
  gmt[slat_name_r] = gmt[src_name_r] + '_' + gmt[lat_name_r]

  gmt = thresholding(gmt, threshold, sim_name)

  if topk is not None:
    col, k = topk
    if k > 0:
      gmt = top_k(gmt, col, k, sim_name)

  if not (global_match is None):
    gm, local = global_match
    if gm == 'sm':
      gmt = stable_marriage(gmt, lat_name_l, lat_name_r, sim_name, local)
    elif gm == 'sbm':
      gmt = simmetric_best_match(gmt, lat_name_l, lat_name_r, sim_name, local)

  return gmt.reindex(columns=[
    src_name_l, lat_name_l, slat_name_l,
    src_name_r, lat_name_r, slat_name_r, sim_name
  ])


def match_from_gmt(
    gmt: pd.DataFrame, self_match=False,
    gat_name='GAT', src_name='SOURCE', slat_name='SLAT'):
  join = pd.merge(gmt, gmt, on=gat_name)
  if self_match:
    # join = join[join[src_name + '_x'] <= join[src_name + '_y']]
    join = join[join[slat_name + '_x'] <= join[slat_name + '_y']]
  else:
    # join = join[join[src_name + '_x'] < join[src_name + '_y']]
    join = join[join[slat_name + '_x'] < join[slat_name + '_y']]
  return join[[slat_name + '_x', slat_name + '_y']].drop_duplicates()
