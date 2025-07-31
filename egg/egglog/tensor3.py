from __future__ import annotations
from egglog import *
from matrix import Matrix
from dim import Dim, egraph
from sum import Sum

class Tensor3(Expr):
    def __init__(self, d1: Dim, d2: Dim, d3: Dim) -> None: ...

    @classmethod
    def named(cls, name: StringLike) -> Tensor3: ...

    def unfold1(self) -> Matrix: ...
    def unfold2(self) -> Matrix: ...
    def unfold3(self) -> Matrix: ...
    def element_at(self, i: i64, j: i64, k: i64) -> f64Like: ...
    def d1(self) -> Dim: ...
    def d2(self) -> Dim: ...
    def d3(self) -> Dim: ...

d1, d2, d3 = vars_("a b c", Dim)
T1, T2 = vars_("T1 T2", Tensor3)
M1, M2 = vars_("M1 M2", Matrix)
egraph.register(
    rewrite(T1.unfold1().nrows()).to(T1.d1()),
    rewrite(T1.unfold1().ncols()).to(T1.d2() * T1.d3()),
    rewrite(T1.unfold2().nrows()).to(T1.d2()),
    rewrite(T1.unfold2().ncols()).to(T1.d1() * T1.d3()),
    rewrite(T1.unfold3().nrows()).to(T1.d3()),
    rewrite(T1.unfold3().ncols()).to(T1.d1() * T1.d2()),
)

i, j, k, l = vars_("i j k l", i64)
egraph.register(
    rewrite((T1.unfold1() @ M1.krao(M2)).element_at(i, l)).to(
        Sum(T1.d2(),
            Sum(T1.d3(),
                T1.element_at(i, j, k) * (M1.element_at(j, l) * M2.element_at(k, l))
            )
        )
    )
)