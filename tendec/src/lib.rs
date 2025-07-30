pub mod matricization;
pub mod misc;
pub mod candecomp;
pub mod tucker;

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array2, ArrayD, ArrayView2, IxDyn};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    use crate::{matricization::{fold, fold_perm, unfold, unfold_perm}, misc::{fnorm, khatri_rao, kronecker, ttm}, tucker::{hooi, hosvd, ttfm}};

    use super::*;

    #[test]
    fn tensor_flattening() {
        let shape = IxDyn(&[2, 3, 4]);
        let tensor = ArrayD::random(shape, Uniform::new(10, 42));

        let flat_tensor: Array2<i32> = unfold(&tensor.view(), 0);
        assert_eq!(flat_tensor.dim().0, 2);
        assert_eq!(flat_tensor.dim().1, 12);
    }

    #[test]
    fn test_unfold_perm_modes() {
        let tensor = array![
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]]
        ].into_dyn();

        let expected = vec![
            array![
                [1., 2., 3., 4.],
                [5., 6., 7., 8.]
            ],
            array![
                [1., 2., 5., 6.],
                [3., 4., 7., 8.]
            ],
            array![
                [1., 5., 3., 7.],
                [2., 6., 4., 8.]
            ],
        ];

        for mode in 0..3 {
            let unfolded = unfold_perm(&tensor.view(), mode);
            assert_eq!(unfolded, expected[mode]);
        }
    }


    #[test]
    fn test_fold_perm_modes() {
        let expected_tensor = array![
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]]
        ].into_dyn();

        let matrices = vec![
            array![
                [1., 2., 3., 4.],
                [5., 6., 7., 8.]
            ],
            array![
                [1., 2., 5., 6.],
                [3., 4., 7., 8.]
            ],
            array![
                [1., 5., 3., 7.],
                [2., 6., 4., 8.]
            ],
        ];

        for mode in 0..3 {
            let folded = fold_perm(&matrices[mode].view(), expected_tensor.shape(), mode);
            assert_eq!(folded, expected_tensor);
        }
    }

    #[test]
    fn test_unfold_fold_perm_roundtrip() {
        let tensor = array![
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]]
        ].into_dyn();

        for mode in 0..3 {
            let unfolded = unfold_perm(&tensor.view(), mode);
            let folded = fold_perm(&unfolded.view(), tensor.shape(), mode);
            assert_eq!(folded, tensor);
        }
    }

    #[test]
    fn kronecker_product_hardcoded() {
        let first_matrix = array![
            [1, 0, 2],
            [0, 1, 1],
            [2, 1, 0]
        ];

        let second_matrix = array![
            [5, 6],
            [7, 8],
            [9, 10]
        ];

        let expected = array![
            [ 5,  6,  0,  0, 10, 12],
            [ 7,  8,  0,  0, 14, 16],
            [ 9, 10,  0,  0, 18, 20],
            
            [ 0,  0,  5,  6,  5,  6],
            [ 0,  0,  7,  8,  7,  8],
            [ 0,  0,  9, 10,  9, 10],
            
            [10, 12,  5,  6,  0,  0],
            [14, 16,  7,  8,  0,  0],
            [18, 20,  9, 10,  0,  0]
        ];

        let result = kronecker(&first_matrix.view(), &second_matrix.view());

        assert_eq!(result.shape(), &[9, 6]);
        assert_eq!(result, expected);
    }

    #[test]
    fn kronecker_product_dimensions() {
        let first_matrix: Array2<f32> = Array2::<f32>::random((3, 3), Uniform::new(0.0, 4.20));
        let second_matrix: Array2<f32> = Array2::<f32>::random((3, 2), Uniform::new(6.9, 66.6));
        let kronecker_product = kronecker(&first_matrix.view(), &second_matrix.view());
        assert_eq!(kronecker_product.dim().0, first_matrix.dim().0*second_matrix.dim().0);
        assert_eq!(kronecker_product.dim().1, first_matrix.dim().1*second_matrix.dim().1);
    }

    #[test]
    fn khatri_rao_hardcoded() {
        use ndarray::array;

        let first_matrix = array![
            [1, 3, 2],
            [1, 0, 0],
            [1, 2, 2],
            [0, 1, 4],
            [3, 0, 1],
        ];

        let second_matrix = array![
            [0, 5, 2],
            [5, 0, 3],
            [1, 1, 1],
        ];

        let expected = array![
            [0, 15, 4],
            [5,  0, 6],
            [1,  3, 2],
            [0,  0, 0],
            [5,  0, 0],
            [1,  0, 0],
            [0, 10, 4],
            [5,  0, 6],
            [1,  2, 2],
            [0,  5, 8],
            [0,  0, 12],
            [0,  1, 4],
            [0,  0, 2],
            [15, 0, 3],
            [3,  0, 1],
        ];

        let result = khatri_rao(&first_matrix.view(), &second_matrix.view());
        assert_eq!(result, expected);
    }

    #[test]
    fn khatri_rao_dimensions() {
        let common_dim: usize = 3;
        let first_matrix: Array2<f32> = Array2::<f32>::random((5, common_dim), Uniform::new(0.0, 4.20));
        let second_matrix: Array2<f32> = Array2::<f32>::random((3, common_dim), Uniform::new(6.9, 66.6));
        let khatri_product = khatri_rao(&first_matrix.view(), &second_matrix.view());
        assert_eq!(khatri_product.dim().0, first_matrix.dim().0*second_matrix.dim().0);
        assert_eq!(khatri_product.dim().1, common_dim);
    }

    fn arrays_approx_equal(a: &ArrayView2<f64>, b: &ArrayView2<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (*x - *y).abs() < tol)
    }

    #[test]
    fn test_pseudoinverse() {
        let a: Array2<f64> = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ];

        let a_pinv = misc::pseudoinverse(&a.view()).expect("Pseudoinverse failed");
        let reconstructed = a.dot(&a_pinv).dot(&a);
        assert!(
            arrays_approx_equal(&a.view(), &reconstructed.view(), 1e-6),
            "A ≉ A * A⁺ * A"
        );
    }

    #[test]
    fn fnorm_known_tensor() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2, 1]), data).unwrap();

        let result = fnorm(&tensor.view());
        let expected = (1.0f64*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0).sqrt();

        let eps = 1e-10;
        assert!((result - expected).abs() < eps, "Expected {}, got {}", expected, result);
    }

    #[test]
    fn test_fold() {
        use ndarray::{array, IxDyn};
        
        let mat = array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ];
        
        let shape = vec![2, 2, 2];
        let folded = fold(&mat.view(), &shape, 0);

        let expected = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                1., 2., 3., 4.,
                5., 6., 7., 8.
            ]
        ).unwrap();

        assert_eq!(folded, expected);
    }

    #[test]
    fn unfold_fold() {
        use ndarray::{Array, IxDyn};

        let original = Array::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![
                1., 2., 3., 4.,
                5., 6., 7., 8.
            ]
        ).unwrap();

        for mode in 0..3 {
            let unfolded = unfold(&original.view(), mode);
            let folded = fold(&unfolded.view(), &[2, 2, 2], mode);

            assert_eq!(original, folded, "Mismatch after unfolding and folding in mode {}", mode);
        }
    }

    #[test]
    fn ttm_mode0() {
        // Create a tensor of shape (2, 3, 4)
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let tensor = Array::from_shape_vec((2, 3, 4), data).unwrap().into_dyn();

        // Matrix of shape (5, 2) to multiply along mode 0 (which has size 2)
        let matrix = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0]
        ];

        // Expected new shape: (5, 3, 4)
        let result = ttm(&tensor.view(), &matrix.view(), 0);

        assert_eq!(result.shape(), &[5, 3, 4]);

        let original_tensor = tensor.into_dimensionality::<ndarray::Ix3>().unwrap();
        let res_tensor = result.into_dimensionality::<ndarray::Ix3>().unwrap();

        for i in 0..3 {
            for j in 0..4 {
                assert_abs_diff_eq!(res_tensor[[0, i, j]], original_tensor[[0, i, j]], epsilon = 1e-8);
                assert_abs_diff_eq!(res_tensor[[1, i, j]], original_tensor[[1, i, j]], epsilon = 1e-8);
                assert_abs_diff_eq!(
                    res_tensor[[2, i, j]],
                    original_tensor[[0, i, j]] + original_tensor[[1, i, j]],
                    epsilon = 1e-8
                );
            }
        }
    }
    
    #[test]
    fn hosvd_roundtrip() {
        use ndarray::array;
        use approx::assert_abs_diff_eq;

        // Create a small 2x3x4 tensor
        let data = array![
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]
        ].into_dyn();

        let ranks = vec![2, 3, 4];
        let (core, factors) = hosvd(&data, &ranks).expect("HOSVD failed");

        let reconstructed = ttfm(&core, &factors);

        let original = data.clone();
        let diff = &original - &reconstructed;
        let frob_norm = fnorm(&diff.view());

        assert_abs_diff_eq!(frob_norm, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hooi_roundtrip() {
        use ndarray::{arr3};
        use approx::assert_abs_diff_eq;

        let tensor = arr3(&[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        .into_dyn();

        let ranks = vec![2, 2, 2];
        let max_iter = 25;
        let tol = 1e-3;

        let (core, factors, iter_cap, conv) = hooi(&tensor, &ranks, max_iter, tol).expect("HOOI failed");
        let approx_tensor = ttfm(&core, &factors);
        assert_eq!(approx_tensor.shape(), tensor.shape());

        for (a, b) in approx_tensor.iter().zip(tensor.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-5);
        }
    }
}