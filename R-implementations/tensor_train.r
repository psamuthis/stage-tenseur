library("rTensor")

tensor_train <- function(tensor, tol = 1e-03) {
    tensor_dims <- dim(tensor)
    dim_count <- length(tensor_dims)
    trunc_param <- (tol/(sqrt(dim_count-1))) * rTensor::fnorm(tensor)
    col_idx <- setdiff(seq_along(tensor_dims), 1)
    C <- unfold(tensor, 1, col_idx)@data
    r = c(1)
    transfer_tensors <- list()

    for (k in seq_len(dim_count-1)) {
        C_rows <- r[k]*tensor_dims[k]
        C_cols <- (nrow(C)*ncol(C))/C_rows
        C <- matrix(C, nrow = C_rows, ncol = C_cols)
	print("iter")
	print(k)
	print("C dim")
	cat(C_rows, "x", C_cols)

        match_rank <- min(nrow(C), ncol(C))
        sing_val <- matrix(nrow = 1, ncol = 1)
        trunc_svd <- matrix(nrow = 1, ncol = 1)
        for (rank in seq_len(min(nrow(C), ncol(C)))) {
            trunc_svd <- svd(C, nu = rank, nv = rank)
            sing_val <- diag(trunc_svd$d[1:rank], nrow=rank, ncol=rank) 
            approx_err <- norm(C - (trunc_svd$u %*% sing_val %*% t(trunc_svd$v)), type="F")

            if(approx_err <= trunc_param) {
                match_rank <- rank
                break 
            }
        }
        r <- append(r, match_rank)
        transfer_tensors[[k]] <- array(trunc_svd$u, dim = c(r[k], tensor_dims[k], r[k + 1]))
	print("tt_dim")
	print(dim(transfer_tensors[[k]]))
        C <- sing_val %*% t(trunc_svd$v)
    }

    #transfer_tensors[[dim_count]] <- C
    transfer_tensors[[dim_count]] <- array(C, dim = c(r[dim_count], tensor_dims[dim_count], 1))
    print("last")
    print(dim(C))
    print(dim(transfer_tensors[[dim_count]]))
    return (transfer_tensors)
}

reconstruct_tt <- function(transfer_tensors) {
  N <- length(transfer_tensors)
  
  # Start with G1
  result <- transfer_tensors[[1]]  # shape: 1 × I1 × r1
  result <- array(result, dim = dim(result))  # make sure it's 3D

  for (k in 2:N) {
    Gk <- transfer_tensors[[k]]
    r_prev <- dim(result)[length(dim(result))]
    Gk_dims <- dim(Gk)

    result_mat <- array(result, dim = c(prod(dim(result)) / r_prev, r_prev))
    Gk_mat <- array(Gk, dim = c(Gk_dims[1], Gk_dims[2] * Gk_dims[3]))
    contracted <- result_mat %*% Gk_mat

    new_dim <- c(dim(result)[-length(dim(result))], Gk_dims[2], Gk_dims[3])
    result <- array(contracted, dim = new_dim)
  }

  final_dims <- dim(result)
  result <- array(result, dim = final_dims[-length(final_dims)])
  return(result)
}

tensor <- rand_tensor(c(3, 4, 2)) #worst case scenario
tt <- tensor_train(tensor)
#for (core in tt) {
  #print(dim(core))
#}
#reconstructed <- reconstruct_tt(tt)
#reconstructed <- as.tensor(array(reconstructed, dim = dim(reconstructed)[-1]))
#print(rTensor::fnorm(tensor))
#print(rTensor::fnorm(reconstructed))
#print(rTensor::fnorm(tensor - reconstructed))
