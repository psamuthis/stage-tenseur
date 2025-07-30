
library("png")
library("rTensor")
source("tensor_train.r")

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

tt <- tensor_train(images_tensor)
for (core in tt) {
  print(dim(core))
}