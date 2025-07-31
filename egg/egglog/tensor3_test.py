from tensor3 import *

x = Dim.named("x")
ten = Dim(10)
res = x * ten * ten
simplified = egraph.simplify(res, 10)
print(simplified)

t = Tensor3(Dim(4), Dim(3), Dim(2))
#t = Tensor3.named("t")
t_unfold1 = t.unfold1()
print(egraph.simplify(t_unfold1.nrows(), 10))
print(egraph.simplify(t_unfold1.ncols(), 10))
t_unfold2 = t.unfold2()
print(egraph.simplify(t_unfold2.nrows(), 10))
print(egraph.simplify(t_unfold2.ncols(), 10))
t_unfold3 = t.unfold3()
print(egraph.simplify(t_unfold3.nrows(), 10))
print(egraph.simplify(t_unfold3.ncols(), 10))

egraph.register(t)
egraph.register(t_unfold1)
egraph.simplify(t_unfold1.ncols(), 10)
egraph.simplify(t_unfold1.nrows(), 10)
#egraph.run(10)
egraph.display()