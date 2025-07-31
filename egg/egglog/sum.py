from __future__ import annotations
from egglog import *
from dim import Dim

class Sum(Expr):
    def __init__(self, range: Dim, expr: f64Like) -> None: ...