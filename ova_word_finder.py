from imports import *
from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
import warnings
from datasets import *

warnings.filterwarnings("ignore")
map = list()

depth=1
for depth range(6):
    t=multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=depth))
    h.train(WineData.X, WineData.Y)
    P = h.predictAll(WineData.Xte)
    ((P == WineDataSmall.Yte),
