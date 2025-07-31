from __future__ import annotations
from egglog import *

egraph = EGraph()

class Dim(Expr):
    def __init__(self, value: i64Like) -> None: ...

    @classmethod
    def named(cls, name: StringLike) -> Dim: ...

    def __mul__(self, other: Dim) -> Dim: ...

a, b, c = vars_("a b c", Dim)
i, j = vars_("i j", i64)
egraph.register(
    rewrite(a * (b * c)).to((a * b) * c),
    rewrite((a * b) * c).to(a * (b * c)),
    rewrite(Dim(i) * Dim(j)).to(Dim(i * j)),
    rewrite(a * b).to(b * a),
)