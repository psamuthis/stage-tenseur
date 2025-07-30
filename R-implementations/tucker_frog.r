library("png")
library("rTensor")
source("tensor_train.r")
source("hooi.r")

file_path <- "/home/rousseau/Documents/datasets/coil-100/"
file_prefix <- "obj28__"
file_ext <- ".png"
data_dim <- c(128, 128, 3, 72)
source_images <- array(0, data_dim)

img_cpt <- 1
for(i in seq(0, 20, by=5)) {
    current_file <- paste0(file_path, file_prefix, i, file_ext)
    source_images[, , , img_cpt] <- readPNG(current_file)
    img_cpt <- img_cpt + 1
}
images_tensor <- as.tensor(source_images)
image <- as.matrix(readPNG(paste0(file_path, file_prefix, "315", file_ext)))

n_ranks <- c(64, 64, 3, 72)
system.time({ mine <- my_hooi(images_tensor, n_ranks, gpu = FALSE) })
system.time({ mine <- my_hooi(images_tensor, n_ranks, gpu = TRUE) })
print(rTensor::fnorm(images_tensor - mine$core_tensor))
print(rTensor::fnorm(mine$core_tensor))

system.time({ theirs <- tucker(images_tensor, n_ranks) })
print(rTensor::fnorm(images_tensor - theirs$est))
print(rTensor::fnorm(theirs$est))