library("rTensor")
library("rje")

matricization <- function(tensor, mode) {
    tensor_dim <- dim(tensor);
    matrix_height <- prod(tensor_dim[-(mode)]);
    matrix <- matrix(nrow=tensor_dim[mode], ncol=matrix_height);
    indices <- arrayInd(1:prod(tensor_dim), tensor_dim);
    
    for (sub_indices in 1:nrow(indices)) {
        current_indices <- indices[sub_indices, ];
        translated_coords <- c(current_indices[mode], toMatrixCol(current_indices, mode, tensor_dim));
        matrix[translated_coords[1], translated_coords[2]] <- do.call(`[`, c(list(tensor@data), as.list(current_indices))); #get the tensor element at current_indices :|
    }

    return (matrix);
}

toMatrixCol <- function(coords, mode, tensor_dim) {
    j <- 1;

    for (idx in 1:length(coords)) {
        if (idx == mode) { next }

        idx_fact <- 1
        if (!(idx == 1 | (idx == 2 & mode == 1))) {
            for (i in 1:(idx-1)) {
                if (i == mode) { next; }
                idx_fact <- idx_fact * tensor_dim[i]
            } 
        }

        j <- j + (coords[idx] - 1) * idx_fact
    }
 
    return (j)
}

#tensor_modes = c(3, 4, 2);
#tensor <- rand_tensor(tensor_modes);

#system.time({matricization(tensor, 1)});

#tensor2 <- rand_tensor(c(7, 3, 10, 2, 6, 1, 4, 4))
#system.time({matricization(tensor2, 3)})
#system.time({unfold(tensor2, 3, c(1,2,4,5,6,7,8))})
