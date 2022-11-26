import numpy as np

from py_schema_matching.utils import concat_dfs
import pandas as pd


def max_sim_table(tables: list, schema: pd.Series = None, sim_col='sim'):
  df, schema = concat_dfs(tables, schema)
  group_by = schema.loc[schema != sim_col].to_list()
  return df.groupby(group_by)[sim_col].max().reset_index()


def min_sim_table(tables: list, schema: pd.Series = None, sim_col='sim'):
  df, schema = concat_dfs(tables, schema)
  group_by = schema.loc[schema != sim_col].to_list()
  return df.groupby(group_by)[sim_col].min().reset_index()


def avg_sim_table(tables: list, schema: pd.Series = None, sim_col='sim'):
  df, schema = concat_dfs(tables, schema)
  group_by = schema.loc[schema != sim_col].to_list()
  return df.groupby(group_by)[sim_col].mean().reset_index()


def wavg_sim_table(tables: list, weights: list, schema: pd.Series = None, sim_col='sim'):
  df, schema = concat_dfs(tables, schema)
  group_by = schema.loc[schema != sim_col].to_list()
  weights = weights / np.sum(weights)
  wavg = lambda x: np.average(x, weights=weights)
  return df.groupby(group_by).agg(**{
    sim_col: (sim_col, wavg)}).reset_index()


def ensemble_sim_table(df0: pd.DataFrame, df1: pd.DataFrame,
                       sim_fns: list, combine,
                       df0_name='df0', df1_name='df1', sim_name='sim'):
  sim_tables = []
  for t in sim_fns:
    f = t[0]
    o = t[1]
    args = {}
    if len(t) > 2:
      args = t[2]
    sim_tables.append(f(o, df0, df1, df0_name=df0_name, df1_name=df1_name,
                        sim_name=sim_name, **args))

  sim_table = []
  if type(combine) is str:
    if combine == 'max':
      sim_table = max_sim_table(sim_tables)
    elif combine == 'min':
      sim_table = min_sim_table(sim_tables)
    elif combine == 'avg':
      sim_table = avg_sim_table(sim_tables)
  elif type(combine) is tuple:
    # TODO: check if lengths match
    sim_table = wavg_sim_table(sim_tables, combine)
  else:
    sim_table, _ = concat_dfs(sim_tables)

  return sim_table
