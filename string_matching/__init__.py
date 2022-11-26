import re
import string

import pandas as pd
import py_stringmatching as sm
import py_stringsimjoin as ssj


class ExtendedJaccard:

  def __init__(self, sim_func, threshold=0.5):
    self.sim_func = sim_func
    self.threshold = threshold

  def get_sim_score(self, set1, set2):
    # the only way I have figured out to make it work with tokenized data
    set1_ = pd.Series(set1).drop_duplicates()
    set2_ = pd.Series(set2).drop_duplicates()
    set1_match = []
    set2_match = []

    for i1 in set1_:
      for i2 in set2_:
        sim = self.sim_func(i1, i2)
        if sim >= self.threshold:
          set1_match.append(i1)
          set2_match.append(i2)

    overlap = set1_match  # or set2_match
    set1_match = pd.Series(set1_match).drop_duplicates()
    set2_match = pd.Series(set2_match).drop_duplicates()
    set1_only = pd.concat([set1_, set1_match]).drop_duplicates(keep=False)
    set2_only = pd.concat([set2_, set2_match]).drop_duplicates(keep=False)

    return len(overlap) / (len(overlap) + len(set1_only) + len(set2_only))


class JaccardSetSimJoin:
  # TODO: generalize SetSimJoin and delegate to subclasses the overlap estimation

  def __init__(self, threshold=0.5):
    self.threshold = threshold

  def get_sim_score(self, set1, set2):
    df0 = pd.DataFrame(set1).drop_duplicates()
    df1 = pd.DataFrame(set2).drop_duplicates()

    if df0.empty | df1.empty:
      return 0.0

    df0.columns = ['df0']
    df1.columns = ['df1']
    overlap = ssj.jaccard_join(
      df0, df1,
      'df0', 'df1',
      'df0', 'df1',
      sm.WhitespaceTokenizer(return_set=True),
      threshold=self.threshold,
      show_progress=False,
      l_out_attrs=['df0'], r_out_attrs=['df1'],
      l_out_prefix='', r_out_prefix=''
    )
    s0 = df0['df0']
    s1 = df1['df1']
    s0_only = s0[~s0.isin(overlap['df0'])].drop_duplicates()
    s1_only = s1[~s1.isin(overlap['df1'])].drop_duplicates()

    return len(overlap) / (len(overlap) + len(s0_only) + len(s1_only))


def string_preprocess(s: str, char: str = string.punctuation, word: list = []):
  if type(s) is str:
    s = s.lower()
    for c in char:
      s = s.replace(c, " ")
    for w in word:
      s = s.replace(w, " ")
  else:
    s = str(s)
  s = re.sub(" +", " ", s)
  return s.strip()


def string_preprocess_tokenize(s: str, char: str = string.punctuation, word: list = [],
                               tokenizer=sm.WhitespaceTokenizer(return_set=True)):
  return tokenizer.tokenize(string_preprocess(s, char, word))
