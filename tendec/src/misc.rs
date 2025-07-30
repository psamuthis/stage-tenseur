use ndarray::{s, Array2, ArrayBase, ArrayD, ArrayView2, ArrayViewD, Dim, OwnedRepr, ScalarOperand};
use ndarray_linalg::{error::LinalgError, Lapack, Scalar, SVD};
use ndarray_rand::rand_distr::num_traits::{Float, One, Zero};
use core::f64;
use std::{cmp::max, fmt::Debug};

use crate::matricization::{fold_perm, unfold_perm};

pub fn kronecker<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where T: Clone + Zero + std::ops::Mul<Output = T> + ScalarOperand, {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let mut result = Array2::<T>::zeros((a_rows * b_rows, a_cols * b_cols));

    for ((i, j), scal) in a.indexed_iter() {
        let mut block = result.slice_mut(s![
            i*b_rows .. (i+1)*b_rows,
            j*b_cols .. (j+1)*b_cols ]);
        block.assign(&(b * scal.clone()));
    }

    result
}

pub fn khatri_rao<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where T: Clone + Zero + std::ops::Mul<Output = T> + ScalarOperand, {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    assert_eq!(a_cols, b_cols, "Mismatch in matrix dimensions: ncols_a {} != ncols_b {}", a_cols, b_cols);
    let mut result = Array2::<T>::zeros((a_rows * b_rows, a_cols));

    for ((i, j), scal) in a.indexed_iter() {
        let mut block = result.slice_mut(s![
            i*b_rows .. (i+1)*b_rows,
            j ]);
        block.assign(&(&b.column(j) * scal.clone()));
    }

    result
}

pub fn pseudoinverse<T>(a: &ArrayView2<T>) -> Result<ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>, LinalgError> 
where T: Scalar + Lapack + Debug + One + Zero, {
    let (u_opt, sigmas, vt_opt) = a.svd(true, true)?;
    let u = u_opt.expect("Missing matrix U");
    let vt = vt_opt.expect("Missing matrix VT");

    if u.is_empty() || vt.is_empty() { return Err(LinalgError::MemoryNotCont) }

    let max_matrix_dim= T::Real::real(max(a.nrows(), a.ncols()));
    let tol: <T as Scalar>::Real = max_matrix_dim * sigmas[0] * T::Real::real(f64::EPSILON);
    let sigma_inv: ArrayBase<OwnedRepr<<T as Scalar>::Real>, Dim<[usize; 1]>> = sigmas
        .iter()
        .map(|&s| if s>tol { T::Real::one()/s } else { T::Real::zero() })
        .collect();

    let mut sigma_inv_matrix = Array2::<T>::zeros((vt.nrows(), u.ncols()));
    for idx in 0..sigma_inv.len() { sigma_inv_matrix[(idx, idx)] = T::from(sigma_inv[idx]).unwrap(); }
    let matrix_inv = vt.t().dot(&sigma_inv_matrix).dot(&u.t());

    return Ok(matrix_inv);
}

pub fn fnorm<T>(tensor: &ArrayViewD<T>) -> f64 
where T: Float, {
    tensor.iter().fold(0.0, |acc, &val| acc + val.to_f64().unwrap() * val.to_f64().unwrap()).sqrt()
}

pub fn ttm<T>(tensor: &ArrayViewD<T>, matrix: &ArrayView2<T>, mode: usize) -> ArrayD<T> 
where T: Clone + Default + Scalar, {
    let unfolded_tensor: Array2<T> = unfold_perm(tensor, mode);
    let result: Array2<T> = matrix.dot(&unfolded_tensor);

    let mut new_shape = tensor.shape().to_vec();
    new_shape[mode] = matrix.shape()[0];
    fold_perm(&result.view(), &new_shape, mode)
}