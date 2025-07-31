from matrix import *

x = Dim.named("x")
ten = Dim(10)
res = x * ten * ten
simplified = egraph.simplify(res, 10)
print(simplified)

x = Matrix.identity(Dim.named("x"))
y = Matrix.identity(Dim.named("y"))
x_mult_y = x @ y
print(egraph.simplify(x_mult_y.nrows(), 10))
print(egraph.simplify(x_mult_y.ncols(), 10))

x_kron_y = x.kron(y)
print(egraph.simplify(x_kron_y.nrows(), 10))
print(egraph.simplify(x_kron_y.ncols(), 10))

a = Matrix(Dim(3), Dim(2))
b = Matrix(Dim(2), Dim(2))
a_mult_b = a @ b
print(egraph.simplify(a_mult_b.nrows(), 10))
print(egraph.simplify(a_mult_b.ncols(), 10))