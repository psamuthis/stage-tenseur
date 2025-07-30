library("rTensor")
source("ttm.r")
source("matricization.r")

my_hosvd <- function(tensor, n_ranks) {
    tensor_dim <- dim(tensor)
    tensor_dim_count <- length(tensor_dim)
    factor_matrices <- list()
    core_tensor <- tensor

    for(n in seq.int(1, tensor_dim_count)) {
        mode_n_svd <- svd(matricization(tensor, n))
        factor_matrices[[n]] <- mode_n_svd$u[, 1:n_ranks[n], drop=FALSE]
        core_tensor <- my_ttm(core_tensor, t(factor_matrices[[n]]), n)
        #core_tensor <- ttm(core_tensor, t(factor_matrices[[n]]), n)
    }

    return (list(
        core_tensor = core_tensor,
        matrices = factor_matrices)
    )
}

tensor_dim = c(100, 100, 50)
tensor = rand_tensor(tensor_dim)
n_ranks <- c(33, 33, 15)
#
#print("my_hosvd")
#system.time({ out <- my_hosvd(tensor, n_ranks) })
#print(rTensor::fnorm(out$core_tensor))
#
#print("rTensor hosvd")
#system.time({ out1 <- hosvd(tensor, n_ranks) })
#print(rTensor::fnorm(out1$est))
