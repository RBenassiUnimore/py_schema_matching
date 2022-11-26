def thresholding(df, thr, sim_name='sim'):
  return df[df[sim_name] > thr].sort_values([sim_name], ascending=False)


def sort_group_cumcount(table, sort_by, group_by):
  return table.sort_values([sort_by], ascending=[False]) \
    .groupby([group_by]).cumcount()


def top_k(df, col, k, sim_name='sim'):
  table_ = df.copy()
  table_['pos'] = sort_group_cumcount(table_, sim_name, col)
  return table_[table_.pos < k].drop(columns='pos').sort_values(
      [col, sim_name], ascending=[True, False])
