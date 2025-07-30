use ndarray::{s, Array2, ArrayD, IxDyn};
use ndarray_linalg::{error::LinalgError,Lapack, Scalar, SVD};
use ndarray_rand::rand_distr::num_traits::Float;

use crate::{matricization::unfold_perm, misc::{fnorm, ttm}};

pub fn hosvd<T>(tensor: &ArrayD<T>, ranks: &Vec<usize>) -> Result<(ArrayD<T>, Vec<Array2<T>>), LinalgError>
where T: Clone + Default + Scalar + Lapack, {
    assert_eq!(tensor.ndim(), ranks.len(), "Number of ranks specified to not match tensor modes.");
    let mut factor_matrices: Vec<Array2<T>> = Vec::<Array2<T>>::with_capacity(ranks.len());

    for rank in 0..ranks.len() {
        let mode_n_matrix: Array2<T> = unfold_perm(&tensor.view(), rank);

        let(u_opt, _, _) = &mode_n_matrix.svd(true, false)?;
        let u: Array2<T> = u_opt.clone().expect("Missing matrix U");
        if u.is_empty() { return Err(LinalgError::MemoryNotCont) }

        factor_matrices.push(u.slice(s![.., 0..ranks[rank]]).to_owned());
    }

    let mut core_tensor: ArrayD<T> = tensor.clone();
    for current_rank in (0..ranks.len()).rev() {
        let u_t: Array2<T> = factor_matrices[current_rank].t().to_owned();
        core_tensor = ttm(&core_tensor.view(), &u_t.view(), current_rank);
    }

    Ok((core_tensor, factor_matrices))
}

pub fn hooi<T>(tensor: &ArrayD<T>, ranks: &Vec<usize>, max_iter: usize, tol: f64) -> Result<(ArrayD<T>, Vec<Array2<T>>, bool, bool), LinalgError>
where T: Default + Lapack + Float, {
    let (_, mut factor_matrices) = hosvd(tensor, ranks).expect("HOSVD failed");
    let mut iter_cnt: usize = 1;
    let mut current_norm = 0.0;
    let mut previous_norm = 0.0;
    let mut convergence: bool = false;

    while !convergence {
        for dim in 0..ranks.len() {
            let mut weight_tensor: ArrayD<T> = ArrayD::<T>::default(IxDyn(&[]));

            for (idx, matrix) in factor_matrices.iter().enumerate().rev().filter(|(idx, _)| *idx != dim) {
                weight_tensor = ttm(&tensor.view(), &matrix.t(), idx);
            }

            let (u_opt, _, _) = unfold_perm(&weight_tensor.view(), dim).svd(true, false)?;
            let u = u_opt.expect("Missing matrix U");
            if u.is_empty() { return Err(LinalgError::MemoryNotCont) }
            factor_matrices[dim] = u.slice(s![.., 0..ranks[dim]]).to_owned();
        }

        current_norm = factor_matrices.iter().map(|m| fnorm(&m.view().into_dyn())).sum();
        convergence = (iter_cnt >= max_iter) || (current_norm - previous_norm < tol);

        iter_cnt += 1;
        previous_norm = current_norm;
    }

    let mut core_tensor: ArrayD<T> = tensor.clone();
    for current_rank in (0..ranks.len()).rev() {
        let u_t: Array2<T> = factor_matrices[current_rank].t().to_owned();
        core_tensor = ttm(&core_tensor.view(), &u_t.view(), current_rank);
    }
    Ok((core_tensor
       ,factor_matrices
       ,(iter_cnt >= max_iter)
       ,(current_norm - previous_norm < tol)
    ))
}

pub fn ttfm<T>(tensor: &ArrayD<T>, factors: &Vec<Array2<T>>) -> ArrayD<T> 
where T: Clone + Default + Scalar + Lapack, {
    let mut reconstructed = tensor.clone();
    for (mode, factor) in factors.iter().enumerate() {
        reconstructed = ttm(&reconstructed.view(), &factor.view(), mode);
    }
    reconstructed
}