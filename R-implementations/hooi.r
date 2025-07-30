source("hosvd.r")
source("matricization.r")
source("ttm.r")
library("rTensor")
library("parallel")


update_tensor <- function(tensor, fact_matrices, skip_idx = NULL, gpu = FALSE) {
    updated_tensor <- tensor
    for(n in rev(seq_len(length(dim(tensor))))) {
        if(!is.null(skip_idx) && n == skip_idx) { next }
        #if(!gpu) { updated_tensor <- my_ttm(tensor, t(fact_matrices[[n]]), n) }
        if(!gpu) { updated_tensor <- ttm(tensor, t(fact_matrices[[n]]), n) }
        else { updated_tensor <- my_ttm_gpu(tensor, t(fact_matrices[[n]]), n) }
        #updated_tensor <- ttm(updated_tensor, t(fact_matrices[[n]]), m=n)
    }

    return (updated_tensor)
}

my_hooi <- function(tensor, n_ranks, max_iter = 25, tol = 1e-05, gpu = FALSE) {
    Un <- my_hosvd(tensor, n_ranks)$matrices
    prev_core <- NULL
    converged <- FALSE
    iter_cnt <- 0

    while(iter_cnt < max_iter && !converged) {
        for(n in 1:length(dim(tensor))) {
            weight_tensor <- update_tensor(tensor, Un, n, gpu)
            mode_n_svd <- svd(matricization(weight_tensor, n))
            Un[[n]] <- mode_n_svd$u[, 1:n_ranks[n]]
        }

        current_core <- update_tensor(tensor, Un, gpu = gpu)
        if(!is.null(prev_core)) {
            converged <- fnorm(current_core - prev_core) < tol
        }
        iter_cnt <- iter_cnt + 1
    }

    return ( list (
        core_tensor = update_tensor(tensor, Un, gpu = gpu),
        factor_matrices = Un,
        converged = converged,
        iter = iter_cnt
    ))
}

n_ranks <- c(100, 100, 50)
tensor <- rand_tensor(c(300, 500, 70))
#
#mine <- my_hooi(tensor, n_ranks)
system.time({theirs <- tucker(tensor, n_ranks)})
#
#rTensor::fnorm(mine$core_tensor)
#rTensor::fnorm(theirs$est)
