import numpy as np
import pandas as pd

from py_schema_matching.data_integration import to_gmm
from py_schema_matching.data_integration.local import local_match_table, local_attribute_table


def transitive_closure(edges):
  closure = set(edges)
  while True:
    new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)
    closure_until_now = closure | new_relations  # UNION
    if closure_until_now == closure:
      break
    closure = closure_until_now
  return closure


def symmetric_closure(edges):
  return set([(b, a) for (a, b) in set(edges)])


def reflexive_closure(nodes):
  return set([(n, n) for n in nodes])


def refl_transitive_closure(nodes, edges=None, edges_n0=None, edges_n1=None):
  if edges is None:
    edges = zip(edges_n0, edges_n1)
  edges = list(edges)  # remove potential iterables
  edges = edges + [(b, a) for a, b in edges]
  c = transitive_closure(edges)
  c.update(symmetric_closure(c))
  c.update(reflexive_closure(nodes))
  return c


### Clusters


def clusters_as_pairs(nodes, edges=None, edges_n0=None, edges_n1=None, gmt=False):
  match_table = pd.DataFrame(
    refl_transitive_closure(nodes, edges, edges_n0, edges_n1),
    columns=['ClusterElement', 'ClusterKey'])
  cluster = match_table.groupby('ClusterElement').agg(
    {'ClusterKey': np.max}).reset_index()
  ret = cluster[['ClusterKey', 'ClusterElement']].sort_values('ClusterKey')

  if gmt:
    ret = ret.rename(columns={'ClusterKey': 'GAT', 'ClusterElement': 'SLAT'})
    ret[['SOURCE', 'LAT']] = ret['SLAT'].str.split('_', expand=True, n=1)

  return ret.reset_index(drop=True)


def clusters(nodes, edges=None, edges_n0=None, edges_n1=None):
  return pd.DataFrame(clusters_as_pairs(nodes, edges, edges_n0, edges_n1).groupby(
    'ClusterKey')['ClusterElement'].apply(list))


def schema_integration(
    dfs: dict, sim_fns: list, combine: str, threshold=0.5,
    top_k: tuple = None, global_match: str = None,
    src_name='SOURCE', lat_name='LAT', slat_name='SLAT',
    suffixes=('_x', '_y'), sim_name='sim'):
  lmt = local_match_table(
    dfs, sim_fns, combine, threshold, top_k, global_match, True,
    src_name, lat_name, slat_name, suffixes, sim_name)
  lat = local_attribute_table(dfs)
  slat_x = slat_name + suffixes[0]
  slat_y = slat_name + suffixes[1]
  gmt = clusters_as_pairs(
    lat[slat_name], edges_n0=lmt[slat_x], edges_n1=lmt[slat_y], gmt=True)
  return to_gmm(gmt)
