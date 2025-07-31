from __future__ import annotations
from egglog import *
from dim import Dim, egraph

class Matrix(Expr):
    def __init__(self, row: Dim, col: Dim) -> None: ...
    @classmethod
    def identity(cls, dim: Dim) -> Matrix: ...
    @classmethod
    def named(cls, name: StringLike) -> Matrix: ...

    def __matmul__(self, other: Matrix) -> Matrix: ...
    def kron(self, b: Matrix) -> Matrix: ...
    def krao(self, b: Matrix) -> Matrix: ...
    def element_at(self, row_idx: i64Like, col_idx: i64) -> f64Like: ...
    def nrows(self) -> Dim: ...
    def ncols(self) -> Dim: ...

a = var("a", Dim)
A, B, C, D = vars_("A B C D", Matrix)
egraph.register(
    # The dimensions of a kronecker product are the product of the dimensions
    rewrite(A.kron(B).nrows()).to(A.nrows() * B.nrows()),
    rewrite(A.kron(B).ncols()).to(A.ncols() * B.ncols()),

    rewrite(A.krao(B).nrows()).to(A.nrows() * B.nrows()),
    rewrite(A.krao(B).ncols()).to(A.ncols()),

    # The dimensions of a matrix multiplication are the number of rows of the first
    # matrix and the number of columns of the second matrix.
    rewrite((A @ B).nrows()).to(A.nrows()),
    rewrite((A @ B).ncols()).to(B.ncols()),

    # The dimensions of an identity matrix are the input dimension
    rewrite(Matrix.identity(a).nrows()).to(a),
    rewrite(Matrix.identity(a).ncols()).to(a),
)