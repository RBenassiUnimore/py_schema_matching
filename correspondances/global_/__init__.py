import pandas as pd
from py_schema_matching.correspondances.local import sort_group_cumcount


def simmetric_best_match(table, col0, col1, sim_name='sim', local=False):
  if local:
    table_ = table.query(col0 + ' != ' + col1)  # check with previous notebooks
  else:
    table_ = table.copy()

  table_['pos0'] = sort_group_cumcount(table_, sim_name, col0)
  table_['pos1'] = sort_group_cumcount(table_, sim_name, col1)

  return table_[(table_.pos0 == 0) & (table_.pos1 == 0)].drop(
    columns=['pos0', 'pos1']).sort_values(sim_name, ascending=False)


def stable_marriage(table: pd.DataFrame, col0: str, col1: str, sim_name='sim',
                    local=False):
  match = pd.DataFrame(columns=table.columns)  # pd.DataFrame(columns=[col0, col1, sim_name])

  if local:
    table_ = table.query(col0 + ' != ' + col1)  # check with previous notebooks
  else:
    table_ = table.copy()

  table_ = table_.sort_values([sim_name], ascending=[False])
  while True:
    candidates = table_.loc[(~table_[col0].isin(match[col0])) &
                            (~table_[col1].isin(match[col1]))]
    if len(candidates) == 0:
      break
    x = candidates.iloc[0, :]
    match = match.append(x, ignore_index=True)
  return match
