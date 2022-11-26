import pandas as pd

from py_schema_matching.string_matching import string_preprocess
from py_schema_matching.utils import cross_join_columns


def to_table(matrix: pd.DataFrame, vals_name):
  return matrix.stack().reset_index(name=vals_name)


def to_matrix(table: pd.DataFrame, index, columns, vals):
  return table.pivot(index=index, columns=columns, values=vals)


def label_sim_table(sim_obj, df0: pd.DataFrame, df1: pd.DataFrame,
                    df0_name='df0', df1_name='df1', sim_name='sim',
                    preprocess=string_preprocess):
  def sim_fn(s: pd.Series):
    return sim_obj.get_sim_score(preprocess(s[0]), preprocess(s[1]))

  table = cross_join_columns(df0, df1, df0_name, df1_name)
  table[sim_name] = table.apply(sim_fn, axis=1)
  return table.sort_values(sim_name, ascending=False)


def value_sim_table(sim_obj, df0: pd.DataFrame, df1: pd.DataFrame,
                    df0_name='df0', df1_name='df1', sim_name='sim',
                    preprocess=string_preprocess):
  def sim_fn(s: pd.Series):
    return sim_obj.get_sim_score(
      df0[s[0]].apply(preprocess).tolist(),
      df1[s[1]].apply(preprocess).tolist()
    )

  table = cross_join_columns(df0, df1, df0_name, df1_name)
  table[sim_name] = table.apply(sim_fn, axis=1)
  return table.sort_values(sim_name, ascending=False)
