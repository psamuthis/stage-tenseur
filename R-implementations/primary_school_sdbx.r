#source("matricization.r")
#source("cp.r")
library("rTensor")

raw_dataset <- read.csv("/home/rousseau/Documents/datasets/primary-school/primaryschool.csv", header=FALSE, sep="\t")
colnames(raw_dataset) <- c("time", "p1", "p2", "p1_class", "p2_class")
dataset <- raw_dataset[1:3]
print("filtered dataset"); head(dataset)

all_nodes <- sort(unique(c(dataset$p1, dataset$p2)))
id_to_index <- setNames(seq_along(all_nodes), all_nodes)
head(id_to_index)
dataset$p1_mapped <- id_to_index[as.character(dataset$p1)]
dataset$p2_mapped <- id_to_index[as.character(dataset$p2)]


bin_size <- 300 
dataset$time_bin <- floor(dataset$time / bin_size)
time_bins <- sort(unique(dataset$time_bin))

matrix_dim <- length(all_nodes) 
time_samples <- sort(unique(dataset$time))
#time_samples_count <- length(time_samples)
time_samples_count <- length(time_bins)

tensor_array <- array(0, dim = c(matrix_dim, matrix_dim, time_samples_count))
print("time_samples"); head(time_samples)


#for (time_sample in seq_len(time_samples_count)) {
for (time_sample in seq_along(time_bins)) {
    #current_time <- time_samples[time_sample]
    current_time <- time_bins[time_sample]
    #edges <- dataset[dataset$time == current_time, ]
    edges <- dataset[dataset$time_bin == current_time, ]

    for (e in seq_len(nrow(edges))) {
        tensor_array[edges$p1_mapped[e], edges$p2_mapped[e], time_sample] <- 1
        tensor_array[edges$p2_mapped[e], edges$p1_mapped[e], time_sample] <- 1
    }
}

tensor <- as.tensor(tensor_array)
print(dim(tensor))

#my_cp <- candecompParafac(tensor, 13, max_iter = 25)
#my_cp$est

#cp <- cp(tensor, 23, max_iter = 25)
tucker <- rTensor::tucker(tensor, c(13, 13, 4))
png("/tmp/factor_matrix_heatmap.png", width=600, height=400)
heatmap(as.matrix(tucker$U[[1]]), Rowv=NA, Colv=NA, col=colorRampPalette(c("blue", "cyan", "green", "yellow", "orange", "red", "purple"))(256), scale="row", 
        xlab="Components", ylab="Rows", main="Factor Matrix Heatmap")
dev.off()

#print(rTensor::fnorm(cp$est))
#print(rTensor::fnorm(my_cp$est))