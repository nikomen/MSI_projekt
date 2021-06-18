from deslib.des.knora_e import KNORAE
from deslib.des import DESKNN
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate
from deslib.des.knora_u import KNORAU
from sklearn.ensemble import BaggingClassifier
from KNORAU_Nasza import KNORA_U_nasz

class Test():

 def test(score_h1, score_h2, score_h3,score_h4,score_h5, n_splits):
  #Parowe testy statystyczne
   clfs = {
   'KNORA-U': KNORAU(),
   'KNORA-E': KNORAE(),
   'DESKNN' : DESKNN(),
   'Bagging': BaggingClassifier(),
   'KNORA-U_Własna': KNORA_U_nasz()
   }
   scores_n = np.zeros((len(clfs), n_splits))
   for i in range(len(clfs)):
    for j in range(n_splits):
        if i == 0:
         scores_n[i, j] = score_h1[j]
        if i == 1:
         scores_n[i, j] = score_h2[j]
        if i == 2:
         scores_n[i, j] = score_h3[j]
        if i == 3:
         scores_n[i, j] = score_h4[j]
        if i == 4:
         scores_n[i, j] = score_h5[j]

   alfa = .05
   t_statistic = np.zeros((len(clfs), len(clfs)))
   p_value = np.zeros((len(clfs), len(clfs)))

   for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(scores_n[i], scores_n[j])

 #t-statystyka oraz p-wartość

   headers = ["KNORA-U","KNORA-E", "DESKNN", "Bagging", "KNORA-U_Własna"]
   names_column = np.array([["KNORA-U"], ["KNORA-E"], ["DESKNN"],['Bagging'],['KNORA-U_Własna']])
   t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
   t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
   p_value_table = np.concatenate((names_column, p_value), axis=1)
   p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
   #print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

   # Przewaga

   advantage = np.zeros((len(clfs), len(clfs)))
   advantage[t_statistic > 0] = 1
   advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
   #print("Advantage:\n", advantage_table)

   #Różnice statystycznie znaczące

   significance = np.zeros((len(clfs), len(clfs)))
   significance[p_value <= alfa] = 1
   significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
   #print("Statistical significance (alpha = 0.05):\n", significance_table)

   #Wynik końcowy analizy statystycznej

   stat_better = significance * advantage
   stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
   print("Statistically significantly better:\n", stat_better_table)