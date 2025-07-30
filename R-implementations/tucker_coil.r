library("png")
library("rTensor")
source("hooi.r")

file <- "/home/rousseau/Documents/datasets/coil-100/obj6__235.png"
img <- readPNG(file)
img_tensor <- as.tensor(img)

n_ranks <- c(16, 16, 3)

system.time({ mine <- my_hooi(img_tensor, n_ranks, gpu = FALSE) })
system.time({ mine <- my_hooi(img_tensor, n_ranks, gpu = TRUE) })
print(rTensor::fnorm(mine$core_tensor))
str(mine$factor_matrices)
core_tensor <- mine$core_tensor@data
#reconstructed <- ttl(mine$core_tensor, mine$factor_matrices, ms = 1:3)
#reconstructed_img <- reconstructed@data
#writePNG(core_tensor, "/home/rousseau/Documents/decomp-impl/img_cpr_output/core_tensor")
#writePNG(reconstructed_img, "/home/rousseau/Documents/decomp-impl/img_cpr_output/reconstructed_img")

system.time({ theirs <- tucker(img_tensor, n_ranks) })
print(rTensor::fnorm(theirs$est))
#writePNG(core_tensor, "/home/rousseau/Documents/decomp-impl/img_cpr_output/their_image")