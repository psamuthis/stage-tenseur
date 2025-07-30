use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, IxDyn};
use ndarray_rand::rand_distr::num_traits::Zero;

pub fn unfold_perm<T: Clone>(tensor: &ArrayViewD<T>, mode: usize) -> Array2<T> {
    let shape = tensor.shape();
    let mut axes: Vec<usize> = (0..shape.len()).collect();
    axes.swap(0, mode);

    let permuted = tensor.view().permuted_axes(axes);

    let contiguous = ArrayD::from_shape_vec(
        permuted.raw_dim(),
        permuted.iter().cloned().collect(),
    ).expect("Failed to create contiguous array");

    let new_shape: (usize, usize) = (
        shape[mode],
        shape.iter().enumerate()
            .filter(|(i, _)| *i != mode)
            .map(|(_, d)| d)
            .product(),
    );

    contiguous.into_shape_with_order(new_shape).unwrap()
}

#[deprecated]
pub fn unfold<T: Clone + Zero>(tensor: &ArrayViewD<T>, mode: usize) -> Array2<T> {
    let shape = tensor.shape();
    let ndim = shape.len();
    let mode_dim = shape[mode];

    let other_dims: Vec<usize> = shape.iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &d)| d)
        .collect();

    let ncols: usize = other_dims.iter().product();
    let mut result = Array2::<T>::zeros((mode_dim, ncols));

    for (idx, val) in tensor.indexed_iter() {
        let row = idx[mode];

        let mut col = 0;
        let mut multiplier = 1;
        for i in (0..ndim).rev() {
            if i == mode { continue; }
            col += idx[i] * multiplier;
            multiplier *= shape[i];
        }

        result[[row, col]] = val.clone();
    }

    result
}

pub fn fold_perm<T: Clone>(mat: &ArrayView2<T>, shape: &[usize], mode: usize) -> ArrayD<T> {
    let mut axes: Vec<usize> = (0..shape.len()).collect();
    axes.swap(0, mode);
    let permuted_shape: Vec<usize> = axes.iter().map(|&i| shape[i]).collect();

    let reshaped = ArrayD::from_shape_vec(
        IxDyn(&permuted_shape),
        mat.iter().cloned().collect(),
    ).expect("Failed to reshape matrix into permuted shape");

    let mut inverse_axes = vec![0; axes.len()];
    for (i, &a) in axes.iter().enumerate() {
        inverse_axes[a] = i;
    }

    reshaped.permuted_axes(inverse_axes)
}

#[deprecated]
pub fn fold<T: Clone>(matrix: &ArrayView2<T>, shape: &[usize], mode: usize) -> ArrayD<T> 
where T: Default, {
    let mut tensor = ArrayD::<T>::default(IxDyn(shape));

    let mode_dim = shape[mode];
    let other_dims: Vec<usize> = shape.iter().enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &d)| d)
        .collect();

    let ncols: usize = other_dims.iter().product();
    assert_eq!(matrix.shape(), &[mode_dim, ncols]);

    for ((row, col), val) in matrix.indexed_iter() {
        let mut idx = vec![0; shape.len()];
        idx[mode] = row;

        let mut tmp = col;
        for (i, &dim) in shape.iter().enumerate().rev() {
            if i == mode { continue; }
            idx[i] = tmp % dim;
            tmp /= dim;
        }

        tensor[IxDyn(&idx)] = val.clone();
    }

    tensor
}
