from egglog import *

from dim import Dim, egraph
from matrix import Matrix
from tensor3 import Tensor3
from sum import Sum

# Declare dimensions
m = Dim(2)
n = Dim(2)
p = Dim(2)
r = Dim(2)

# Create symbolic tensor and matrices
X = Tensor3.named("X")
B = Matrix.named("B")
C = Matrix.named("C")

# Declare shapes for sanity (optional checks can be added later)
egraph.let("X", Tensor3(m, n, p))
egraph.let("B", Matrix(n, r))
egraph.let("C", Matrix(p, r))

# Define expression U = unfold1(X) @ (B.krao(C))
U = X.unfold1() @ B.krao(C)
U_expr = egraph.let("U_expr", U)

# Define elementwise version:
i, j, k, l = vars_("i j k l", i64)
U_elementwise = Sum(n,
                    Sum(p,
                        X.element_at(i, j, k) * B.element_at(j, l) * C.element_at(k, l)
                    )
                )
U_manual_expr = egraph.let("U_manual_expr", U_elementwise)

# Define a check
egraph.check(U_expr.element_at(i, l) == U_manual_expr)

# Run saturation + extraction
egraph.saturate()
result = egraph.extract(U_expr.element_at(i, l))
print("Result:", result)