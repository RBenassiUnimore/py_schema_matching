import pandas as pd


def concat_dfs(tables: list, schema: pd.Series = None):
  if schema is None:
    schema = tables[0].columns.to_series()
  df = pd.DataFrame(columns=schema)
  for t in tables:
    df = df.append(t[schema], ignore_index=True)
  return df, schema


def cross_join_columns(df0 : pd.DataFrame, df1 : pd.DataFrame, df0_name='df0', df1_name='df1'):
  c0 = pd.DataFrame({df0_name : df0.columns})
  c1 = pd.DataFrame({df1_name : df1.columns})
  return c0.join(c1, how='cross')
