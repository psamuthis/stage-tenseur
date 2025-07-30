mpinv <- function(matrix) {
    tol <- 1e-05
    mat_svd <- svd(matrix)
    sig_len <- length(mat_svd$d)
    sig_inv <- c()

    for (sval in mat_svd$d) {
        if(sval <= tol) {
            sig_inv <- c(sig_inv, 0)
        } else {
            sig_inv <- c(sig_inv, 1/sval)
        }
    }
    sig_inv <- matrix(sig_inv, nrow = sig_len, ncol = sig_len)
    return (mat_svd$v %*% t(sig_inv) %*% t(mat_svd$u))
}

#inv <- mpinv(matrix(runif(6), nrow=3, ncol=2))
#print(inv)