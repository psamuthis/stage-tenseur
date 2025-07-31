from __future__ import annotations
from egglog import *
from sum import Sum
from matrix import Matrix
from dim import Dim

A, B = ("A B", Matrix)
C = Matrix.named("C")
x = f64(3.0)
sum = Sum(Dim(4), x)
print(type(sum))