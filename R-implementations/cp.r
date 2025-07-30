library("rTensor")
library("pracma")
library("rlist")
library("matrixcalc")
source("matricization.r")

init_matrices <- function(tensor_dim, est_components) {
  matrices <- list()

  for (i in 1:length(tensor_dim)) {
    matrices[[i]] <- matrix(runif(tensor_dim[i] * est_components), nrow=tensor_dim[i], ncol=est_components)
  }

  return (matrices)
}

khatriRaoCW <- function(first_matrix, second_matrix) {
    if(ncol(first_matrix) != ncol(second_matrix)) {
        print("dimension mismatch")
        return ()
    }

    result_matrix <- matrix(ncol=ncol(first_matrix), nrow=nrow(first_matrix) * nrow(second_matrix))
    for (i in 1:ncol(result_matrix)) {
        result_matrix[, i] <- kronecker(first_matrix[, i], second_matrix[, i])
    }

    return (result_matrix)
}

computeHadamard <- function(factor_matrices) {
    matrices <- list()

    for (i in 1:length(factor_matrices)) {
        matrices[[i]] <- t(factor_matrices[[i]]) %*% factor_matrices[[i]]
    }

    #return (Reduce(`*`, matrices))
    return (hadamard_list(matrices))
}

computeKhatriRao <- function(factor_matrices) {
    result_matrix <- factor_matrices[[length(factor_matrices)]]

    for(mat in (length(factor_matrices)-1):1) {
        result_matrix <- khatriRaoCW(result_matrix, factor_matrices[[mat]])
    }

    return (result_matrix)
}

candecompParafac <- function(tensor, est_components, max_iter = 25, tol = 1e-05) {
    iter_cnt <- 0
    tensor_dim <- dim(tensor)
    tensor_dim_count <- length(tensor_dim)
    factor_matrices <- init_matrices(tensor_dim, est_components)
    approx_tensor <- rand_tensor(tensor_dim)
    matrices_norms <- rep(1, est_components)
    fnorm_resid <- rTensor::fnorm(tensor - approx_tensor)
    all_modes <- as.numeric(seq_len(tensor_dim_count))

    while(fnorm_resid > tol & iter_cnt < max_iter) {

        for (n in seq_len(tensor_dim_count)) {
            V <- computeHadamard(factor_matrices[-n])

            khatri_product <- computeKhatriRao(factor_matrices[-n])
            #factor_matrices[[n]] <- matricization(tensor, n) %*% (khatri_product %*% pinv(V))
	        unfolded_tensor <- as.matrix(unfold(tensor, n, all_modes[-n])@data)
            factor_matrices[[n]] <-unfolded_tensor %*% (khatri_product %*% pinv(V))
        }

        approx_tensor <- k_fold((factor_matrices[[1]] %*% t((computeKhatriRao(factor_matrices[-1])))), 1, c(tensor_dim))
        fnorm_resid <- rTensor::fnorm(tensor - approx_tensor)
        iter_cnt <- iter_cnt + 1
    }

    return (list(
        lambda = matrices_norms,
        U = factor_matrices,
        fnorm_resid = fnorm_resid,
        est = approx_tensor
    ))
}

reconstruct_cp <- function(lambdas, factor_matrices) {
    dim <- c()
    rank <- dim(factor_matrices[[1]])[2]

    for(i in seq_along(factor_matrices)) {
        dim <- append(dim, dim(factor_matrices[[i]])[1])
    }

    est <- array(0, dim=dim)
    for(r in 1:rank) {
        vec <- factor_matrices[[1]][, r]
        tensor_r <- vec

        for (n in 2:length(dim)) {
            vec_next <- factor_matrices[[n]][, r]
            tensor_r <- outer(tensor_r, vec_next)
        }
        lbd <- as.numeric(lambdas[r])
        est <- est + lbd * array(tensor_r, dim = dim)
    }

    return (est)
}

#est_components <- 10 
#tensor <- rand_tensor(c(100, 100, 50))
#print(rTensor::fnorm(tensor))
#
#print("CP (mine)")
#my_cp <- candecompParafac(tensor, est_components)
#print(my_cp$lambda)
#print(my_cp$U)
#print(my_cp$fnorm_resid)
#print(my_cp$est)

#print("CP (rTensor)")
#cp <- cp(tensor, est_components)
#print(cp$lambda)
#print(cp$U)
#print(cp$fnorm_resid)
#print(rTensor::fnorm(cp$est))