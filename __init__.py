import pandas as pd


def evaluate(gold: pd.DataFrame, df: pd.DataFrame, columns: list = ['GAT', 'SLAT'],
             return_tps=False, return_fps=False, return_fns=False):
  # FOJ = gold[columns].merge(df[columns], how='outer', indicator=True)
  FOJ = pd.merge(gold, df, how='outer', indicator=True, on=columns)
  TP = FOJ[FOJ['_merge'] == 'both']
  FP = FOJ[FOJ['_merge'] == 'right_only']
  FN = FOJ[FOJ['_merge'] == 'left_only']

  P = 0
  R = 0
  F = 0

  if len(TP) > 0:
    P = len(TP) / (len(TP) + len(FP))
    R = len(TP) / (len(TP) + len(FN))
    F = 2 * P * R / (P + R)

  ret = [pd.DataFrame({
    'MT': [len(df)],
    'TP': [len(TP)],
    'FP': [len(FP)],
    'FN': [len(FN)],
    'P': [round(P, 4)],
    'R': [round(R, 4)],
    'F': [round(F, 4)]
  })]
  if return_tps: ret.append(TP)
  if return_fps: ret.append(FP)
  if return_fns: ret.append(FN)
  return tuple(ret) if len(ret) > 1 else ret[0]


def is_functnl_dep(df: pd.DataFrame, attr1, attr2, ret_dep_violtns=False):
  dep_violtns = df.groupby(attr1)[attr2].nunique().reset_index(
    name='counts').query('counts>1')
  ret = [dep_violtns.shape[0]]
  if ret_dep_violtns:
    ret.append(dep_violtns)
  return ret if len(ret) > 1 else ret[0]


def get_mapping(df, gat_name='GAT', src_name='SOURCE', lat_name='LAT', slat_name='SLAT'):
  n0 = is_functnl_dep(df, [src_name, lat_name], gat_name)
  n1 = is_functnl_dep(df, [src_name, gat_name], slat_name)
  if n0 == 0:
    return '1-1' if n1 == 0 else '1-N'
  if n0 == 1:
    return 'N-1' if n1 == 0 else 'N-N'
