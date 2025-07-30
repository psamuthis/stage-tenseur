use ndarray::{Array1, Array2, ArrayD, IxDyn, LinalgScalar, ScalarOperand};
use ndarray_linalg::{Lapack};
use ndarray_rand::rand_distr::num_traits::{Float, One, Zero};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fmt::Debug;

use crate::{matricization::{unfold, unfold_perm}, misc::{fnorm, khatri_rao, pseudoinverse}};


pub fn cp<T>(tensor: &ArrayD<T>, nrank: usize, max_iter: usize, tol: f64) -> (Vec<Array2<T>>, bool, bool)
where T: Clone + Sync + Send + One + LinalgScalar + ScalarOperand + Lapack + Default + Debug + Float, {
    let tensor_dim = tensor.shape();
    let mut factor_matrices: Vec<Array2<T>> = Vec::with_capacity(nrank);
    let mut iter_cpt = 0;
    let mut current_norm: f64 = 0.0;
    let mut previous_norm: f64 = 0.0;
    let mut convergence: bool = false;

    for mat in 0..tensor_dim.len() {
        let matrix_dim = (tensor_dim[mat], nrank);
        factor_matrices.push(Array2::<T>::ones(matrix_dim));
    }

    while !convergence {
        for current_rank in 0..tensor_dim.len() {
            let hadamard_inv = pseudoinverse(&compute_hadamard(&factor_matrices, current_rank, nrank).view())
                .expect("Error while computing Kronecker");
            let khatri_rao = compute_khatri_rao(&factor_matrices, current_rank);
            factor_matrices[current_rank] = unfold_perm(&tensor.view(), current_rank).dot(&khatri_rao).dot(&hadamard_inv);
        }

        current_norm = factor_matrices.iter().map(|m| fnorm(&m.view().into_dyn())).sum();
        convergence = iter_cpt >= max_iter || (current_norm - previous_norm < tol);
        iter_cpt += 1;

        previous_norm = current_norm;
    } 

    (factor_matrices, iter_cpt >= max_iter, (current_norm - previous_norm < tol))
}

#[inline]
fn compute_hadamard<T>(factor_matrices: &[Array2<T>], current_rank: usize, nrank: usize) -> Array2<T> 
where T: Clone + Sync + Send + One + LinalgScalar + ScalarOperand + Lapack, {
    factor_matrices
        .par_iter()
        .enumerate()
        .filter(|(dim, _)| *dim != current_rank)
        .map(|(_, mat)| mat.t().dot(mat))
        .reduce(
            || Array2::<T>::ones((nrank, nrank)),
            |a, b| a * b,
        ) 
}

#[inline]
fn compute_khatri_rao<T>(factor_matrices: &Vec<Array2<T>>, current_rank: usize) -> Array2<T>
where T: Clone + One + Zero + ScalarOperand + Debug, {
    let mut result_matrix: Option<Array2<T>> = None;

    for (idx, matrix) in factor_matrices.iter().enumerate() {
        if idx == current_rank {
            continue;
        }

        result_matrix = Some(match result_matrix {
            Some(ref mat) => khatri_rao(&mat.view(), &matrix.view()),
            None => matrix.clone(),
        });
    }

    result_matrix.expect("At least one matrix required for Khatri-Rao")
}

pub fn reconstruct_cp_tensor<T>(factors: &[Array2<T>]) -> ArrayD<T>
where T: Clone + Zero + std::ops::Mul<Output = T> + std::ops::AddAssign + One, {
    let rank = factors[0].ncols();
    let shape: Vec<usize> = factors.iter().map(|m| m.nrows()).collect();
    let mut tensor = ArrayD::<T>::zeros(IxDyn(&shape));

    for r in 0..rank {
        let mut rank_one = ArrayD::from_elem(IxDyn(&[]), T::one());
        for factor in factors {
            let vec = factor.column(r).to_owned(); // shape: (dim,)
            rank_one = outer_product_expand(&rank_one, &vec);
        }
        tensor += &rank_one;
    }

    tensor
}

fn outer_product_expand<T>(tensor: &ArrayD<T>, vec: &Array1<T>) -> ArrayD<T>
where
    T: Clone + Zero + std::ops::Mul<Output = T>,
{
    let mut new_shape = tensor.shape().to_vec();
    new_shape.push(vec.len());

    let mut result = ArrayD::<T>::zeros(IxDyn(&new_shape));

    for (i, v) in vec.iter().enumerate() {
        let scaled = tensor.mapv(|x| x.clone() * v.clone());
        let mut slice = result.index_axis_mut(ndarray::Axis(result.ndim() - 1), i);
        slice.assign(&scaled);
    }

    result
}