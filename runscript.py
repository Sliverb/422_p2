from imports import *
from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
from datasets import *

#h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=1))
#h.train(WineData.X, WineData.Y)
#P = h.predictAll(WineData.Xte)
#mean(P == WineData.Yte)
#mode(WineData.Y)
#WineData.labels[1]

#mean(WineData.Yte == 1)

#P = h.predictAll(WineData.Xte, useZeroOne=True)
#mean(P == WineData.Yte)

#h = multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=3))
#h.train(WineDataSmall.X, WineDataSmall.Y)
#P = h.predictAll(WineDataSmall.Xte)
#mean(P == WineDataSmall.Yte)
#mean(WineDataSmall.Yte == 1)

# WineDataSmall.labels[0]
# The 1s mean "likely to be Sauvignon-Blanc" and the 0s mean "likely not to be".
#util.showTree(h.f[0], WineDataSmall.words)

# WineDataSmall.labels[0]
# The 1s mean "likely to be Sauvignon-Blanc" and the 0s mean "likely not to be".
# util.showTree(h.f[0], WineDataSmall.words)

# WineDataSmall.labels[2]
# The 1s mean "likely to be Pinot-Noir" and the 0s mean "likely not to be".
# util.showTree(h.f[2], WineDataSmall.words)

"""
h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
mean(P == WineData.Yte)
mean(WineData.Yte == 1)

t = multiclass.makeBalancedTree(range(6))
t
t.isLeaf
t.getLeft()
t.getLeft().getLeft()
t.getLeft().getLeft().isLeaf

t = multiclass.makeBalancedTree(range(5))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineDataSmall.X, WineDataSmall.Y)
P = h.predictAll(WineDataSmall.Xte)
mean(P == WineDataSmall.Yte)

t = multiclass.makeBalancedTree(range(5))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
mean(P == WineData.Yte)

# implement a generic gradiant decent method 
# In each iteration, we will compute the gradient and take a step in that direction
# with the step size eta 
# We will have an adaptive step size, where eta is computed as stepSize divided by 
# the square root of the iteration number (counted from one)
"""

"""
import gd
x = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)
x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0.2)
print x
plot(trajectory)
close()
show(False)

x, trajectory = gd.gd(lambda x: linalg.norm(x)**2, lambda x: 2*x, array([10,5]), 100, 0.2)
x
plot(trajectory)
"""
# implement LogisticLoss and HingeLoss functions in Linear.py
# LinearClassifier class is a stub implementation of a generic linear classifer 
import linear

f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 0, 'numIter': 100, 'stepSize': 0.5})

runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
"""
f
mlGraphics.plotLinearClassifier(f, datasets.TwoDAxisAligned.X, datasets.TwoDAxisAligned.Y)
show(False)
"""
# note that the unbiased classifier is unable to perfectly seperate the data
"""
f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDAxisAligned)
f

# weights are smaller 
# test out hinge loss and logisitc loss functions

f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 10, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
f

f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
f
"""
input("Press Enter to continue...")