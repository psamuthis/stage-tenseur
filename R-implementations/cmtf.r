library("rTensor")
source("cp.r")
source("pseudoinverse.r")

cmtf <- function(tensor, coupled_matrices, rank, max_iter = 25) {
    stopifnot(length(coupled_matrices) <= length(dim(tensor)))
    decomp <- cp(tensor, rank)
    tensor_fm <- decomp$U
    lambdas <- decomp$lambdas
    mat_fm <- list()
    iter_cnt <- 0

    for(i in seq_along(coupled_matrices)) {
        if(is.null(coupled_matrices[[i]])) { 
            mat_fm <- append(mat_fm, list(NULL))
        } else {
            fm <- t(coupled_matrices[[i]]) %*% t(mpinv(tensor_fm[[i]]))
            mat_fm <- append(mat_fm, list(fm))
        }
    }

    while(iter_cnt < max_iter) {
        for(i in seq_along(coupled_matrices)) {
            if(is.null(coupled_matrices[[i]])) {
                unfold <- unfold(tensor, i, setdiff(seq_along(dim(tensor)), i))@data
                factor <- t(mpinv(khatri_rao_list(tensor_fm[-i])))
                tensor_fm[[i]] <- unfold %*% factor
            } else {
                unfold <- unfold(tensor, i, setdiff(seq_along(dim(tensor)), i))@data
                data <- cbind(unfold, coupled_matrices[[i]])
                factor <- t(mpinv(rbind(khatri_rao_list(tensor_fm[-i]), mat_fm[[i]])))

                tensor_fm[[i]] <- data %*% factor 
                mat_fm[[i]] <- t(coupled_matrices[[i]]) %*% t(mpinv(tensor_fm[[i]]))
            }
        }

        iter_cnt <- iter_cnt + 1
    }

    return (list(
        tensor_fm = tensor_fm,
        mat_fm = mat_fm,
        lambdas = lambdas
    ))
}

#tensor_dim <- c(4, 3, 2)
#tensor <- rand_tensor(tensor_dim)
#cm1 <- matrix(runif(20), nrow=4, ncol=5)
#cm2 <- matrix(runif(12), nrow=3, ncol=4)
#cm3 <- matrix(runif(18), nrow=2, ncol=9)
#rank <- 2

#coupled_matrices <- list(cm1, NULL, NULL)
#decomp <- cmtf(tensor, coupled_matrices, rank)
#rec_ten <- reconstruct_cp(decomp$lambdas, decomp$tensor_fm)
#rec_ten <- as.tensor(rec_ten)
#print(rTensor::fnorm(tensor - rec_ten))
#rec_mat <- decomp$tensor_fm[[1]] %*% t(decomp$mat_fm[[1]])
#print(norm(cm1 - rec_mat, type="F"))

#coupled_matrices <- list(NULL, cm2, NULL)
#decomp <- cmtf(tensor, coupled_matrices, rank)
#rec_ten <- reconstruct_cp(decomp$lambdas, decomp$tensor_fm)
#rec_ten <- as.tensor(rec_ten)
#print(rTensor::fnorm(tensor - rec_ten))
#rec_mat <- decomp$tensor_fm[[2]] %*% t(decomp$mat_fm[[2]])
#print(norm(cm2 - rec_mat, type="F"))

#coupled_matrices <- list(NULL, NULL, cm3)
#decomp <- cmtf(tensor, coupled_matrices, rank)
#rec_ten <- reconstruct_cp(decomp$lambdas, decomp$tensor_fm)
#rec_ten <- as.tensor(rec_ten)
#print(rTensor::fnorm(tensor - rec_ten))
#rec_mat <- decomp$tensor_fm[[3]] %*% t(decomp$mat_fm[[3]])
#print(norm(cm3 - rec_mat, type="F"))