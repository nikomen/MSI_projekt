from deslib.des.knora_e import KNORAE
from deslib.des import DESKNN
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from deslib.des.knora_u import KNORAU
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from Testy_statystyczne import Test
from KNORAU_Nasza import KNORA_U_nasz
#Generowanie danych
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=5, random_state=7)

#Można wczytać dane z pliku
dataset = 'german'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

#Deklaracja klasyfikatorów bazowych
classifiers = [
    DecisionTreeClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    KNeighborsClassifier()
]
#Ilośc foldów w walidacji
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1410)

#zmienne na wyniki
scores1 = np.zeros(n_splits)
scores2 = np.zeros(n_splits)
scores3 = np.zeros(n_splits)
scores4 = np.zeros(n_splits)
scores5 = np.zeros(n_splits)
scores1rd = np.zeros(n_splits)
scores2rd = np.zeros(n_splits)
scores3rd = np.zeros(n_splits)
scores4rd = np.zeros(n_splits)
scores5rd = np.zeros(n_splits)
scores1vm = np.zeros(n_splits)
scores2vm = np.zeros(n_splits)
scores3vm = np.zeros(n_splits)
scores4vm = np.zeros(n_splits)
scores5vm = np.zeros(n_splits)

#walidacja krzyżowa danych
for fold_id, (train, test) in enumerate(kf.split(X, y)):
    for c in classifiers:
        #Uczenie klasyfiaktorów bazowych
        c.fit(X[train], y[train])
        #Weryfikacja ich jakości
        # y_pred = c.predict(X[test])
        # score = accuracy_score(y[test],y_pred)
        #print('>%s: %.3f' % (c.__class__.__name__, score))

#Przypasnie komitetu klasyfikatorów
    model1 = KNORAU(pool_classifiers=classifiers,random_state=1410)
    model2 = KNORAE(pool_classifiers=classifiers,random_state=1410)
    model3 = DESKNN(pool_classifiers=classifiers,random_state=1410)
    model4 = BaggingClassifier()
    model5 = KNORA_U_nasz(pool_classifiers=classifiers, random_state=1410)
    #Uczenie
    model1.fit(X[train], y[train])
    model2.fit(X[train], y[train])
    model3.fit(X[train], y[train])
    model4.fit(X[train], y[train])
    model5.fit(X[train], y[train])
    #Predykcja
    y1 = model1.predict(X[test])
    y2 = model2.predict(X[test])
    y3 = model3.predict(X[test])
    y4 = model4.predict(X[test])
    y5 = model5.predict(X[test])
    #Zapis wyników/ różne metryki
    scores1[fold_id] = accuracy_score(y[test], y1)
    scores2[fold_id] = accuracy_score(y[test], y2)
    scores3[fold_id] = accuracy_score(y[test], y3)
    scores4[fold_id] = accuracy_score(y[test], y4)
    scores5[fold_id] = accuracy_score(y[test], y5)
    scores1rd[fold_id] = f1_score(y[test],y1)
    scores2rd[fold_id] = f1_score(y[test], y2)
    scores3rd[fold_id] = f1_score(y[test], y3)
    scores4rd[fold_id] = f1_score(y[test], y4)
    scores5rd[fold_id] = f1_score(y[test], y5, zero_division=0)
    scores1vm[fold_id] = precision_score(y[test], y1)
    scores2vm[fold_id] = precision_score(y[test], y2)
    scores3vm[fold_id] = precision_score(y[test], y3)
    scores4vm[fold_id] = precision_score(y[test], y4)
    scores5vm[fold_id] = precision_score(y[test], y5,zero_division=0)

#Liczenie średnich
mean_score1 = np.mean(scores1)
std_score1 =np.std(scores1)
mean_score2 = np.mean(scores2)
std_score2 =np.std(scores2)
mean_score3 = np.mean(scores3)
std_score3 =np.std(scores3)
mean_score4 = np.mean(scores4)
std_score4 =np.std(scores4)
mean_score5 = np.mean(scores5)
std_score5 =np.std(scores5)
#Wyniki
print("accuracy score")
print("KNORA-U: %.3f (%.3f)" % (mean_score1, std_score1))
print("KNORA-E: %.3f (%.3f)" % (mean_score2, std_score2))
print("DESKNN: %.3f (%.3f)" % (mean_score3, std_score3))
print("Bagging: %.3f (%.3f)" % (mean_score4, std_score4))
print("KNORA-U_Własna: %.3f (%.3f)" % (mean_score5, std_score5))
#TO co wyżej dla innych metryk
mean_score1 = np.mean(scores1rd)
std_score1 =np.std(scores1rd)
mean_score2 = np.mean(scores2rd)
std_score2 =np.std(scores2rd)
mean_score3 = np.mean(scores3rd)
std_score3 =np.std(scores3rd)
mean_score4 = np.mean(scores4rd)
std_score4 =np.std(scores4rd)
mean_score5 = np.mean(scores5rd)
std_score5 =np.std(scores5rd)
print("f1 score")
print("KNORA-U: %.3f (%.3f)" % (mean_score1, std_score1))
print("KNORA-E: %.3f (%.3f)" % (mean_score2, std_score2))
print("DESKNN: %.3f (%.3f)" % (mean_score3, std_score3))
print("Bagging: %.3f (%.3f)" % (mean_score4, std_score4))
print("KNORA-U_Własna: %.3f (%.3f)" % (mean_score5, std_score5))
mean_score1 = np.mean(scores1vm)
std_score1 =np.std(scores1vm)
mean_score2 = np.mean(scores2vm)
std_score2 =np.std(scores2vm)
mean_score3 = np.mean(scores3vm)
std_score3 =np.std(scores3vm)
mean_score4 = np.mean(scores4vm)
std_score4 =np.std(scores4vm)
mean_score5 = np.mean(scores5vm)
std_score5 =np.std(scores5vm)
print("precision score")
print("KNORA-U: %.3f (%.3f)" % (mean_score1, std_score1))
print("KNORA-E: %.3f (%.3f)" % (mean_score2, std_score2))
print("DESKNN: %.3f (%.3f)" % (mean_score3, std_score3))
print("Bagging: %.3f (%.3f)" % (mean_score4, std_score4))
print("KNORA-U_Własna: %.3f (%.3f)" % (mean_score5, std_score5))
#Testy statystyczne
t = Test
print("Test dla accuracy score")
t.test(scores1,scores2,scores3,scores4,scores5, n_splits)
print("Test dla f1 score")
t.test(scores1rd,scores2rd,scores3rd,scores4rd,scores5rd,n_splits)
print("Test dla precision score")
t.test(scores1vm,scores2vm,scores3vm,scores4vm,scores5vm,n_splits)

#https://deslib.readthedocs.io/en/latest/auto_examples/plot_using_instance_hardness.html#sphx-glr-auto-examples-plot-using-instance-hardness-py
#https://www.researchgate.net/publication/339672014_Data_Preprocessing_for_des-knn_and_Its_Application_to_Imbalanced_Medical_Data_Classification
#https://github.com/scikit-learn-contrib/DESlib
#https://github.com/makonwencjusz/praca_magisterska/blob/master/runnn.py
#https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/
#https://deslib.readthedocs.io/en/latest/user_guide/tutorial.html
#https://deslib.readthedocs.io/en/latest/auto_examples/example_heterogeneous.html
#https://scikit-learn.org/stable/modules/model_evaluation.html
#https://books.google.pl/books?id=IUkrEAAAQBAJ&pg=PA107&lpg=PA107&dq=KNORAU+pool+classifier&source=bl&ots=t-4bbfDeqs&sig=ACfU3U3f1r5_mZNILaEb6zQFDIBwyyZh-g&hl=pl&sa=X&ved=2ahUKEwj50pDqo5zxAhVimIsKHUK9BU0Q6AEwBXoECAoQAw#v=onepage&q=KNORAU%20pool%20classifier&f=false
