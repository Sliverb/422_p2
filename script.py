from imports import *
from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
import warnings
from datasets import *

warnings.filterwarnings("ignore")
h = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
print(mean(WineData.Yte==P))
