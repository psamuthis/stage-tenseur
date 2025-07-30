my_ttm <- function(tensor, mat, mode) {
    mat_tensor <- matricization(tensor, mode)
    res_tensor <- mat %*% mat_tensor
    new_modes <- dim(tensor)
    new_modes[mode] <- nrow(mat)

    return(fold(
        res_tensor,
        row_idx=mode,
        col_idx=setdiff(seq_along(dim(tensor)), mode),
        modes=new_modes
    ))
}

my_ttm_gpu <- function(tensor, mat, mode) {
  library(gpuR)
  
  mat_tensor <- matricization(tensor, mode)
  
  mat_gpu <- vclMatrix(mat, type = "float")
  mat_tensor_gpu <- vclMatrix(mat_tensor, type = "float")
  
  res_gpu <- mat_gpu %*% mat_tensor_gpu
  res_cpu <- as.matrix(res_gpu)
  
  new_modes <- dim(tensor)
  new_modes[mode] <- nrow(mat)
  
  fold(res_cpu, row_idx = mode, col_idx = setdiff(seq_along(dim(tensor)), mode), modes = new_modes)
}