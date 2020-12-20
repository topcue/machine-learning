## Set Up Directory
## CNN HanGul
#############################################

setwd("/Users/topcue/myR")
rm(list=ls())

# install.packages("keras")
# install.packages("tensorflow")
# tensorflow::install_tensorflow()

############## Set Up Directory ##############
## init dir
base_dir <- "/Users/topcue/myR/CNN_Hangul_Data"
original_dataset_dir <- "/Users/topcue/myR/CNN_Hangul_Data/Hangul5_Data"
my_dataset_dir <- file.path(base_dir, "my_Hangul5_Data")

## mkdir my data set directory
my_ga_dir <- file.path(my_dataset_dir, "my_Ga")
my_na_dir <- file.path(my_dataset_dir, "my_Na")
my_da_dir <- file.path(my_dataset_dir, "my_Da")
my_ra_dir <- file.path(my_dataset_dir, "my_Ra")
my_ma_dir <- file.path(my_dataset_dir, "my_Ma")

dir.create(my_dataset_dir)
dir.create(my_ga_dir)
dir.create(my_na_dir)
dir.create(my_da_dir)
dir.create(my_ra_dir)
dir.create(my_ma_dir)

## copy original data set to my data set
original_ga_dir <- file.path(original_dataset_dir, "AC00_Ga")
original_na_dir <- file.path(original_dataset_dir, "B098_Na")
original_da_dir <- file.path(original_dataset_dir, "B2E4_Da")
original_ra_dir <- file.path(original_dataset_dir, "B77C_Ra")
original_ma_dir <- file.path(original_dataset_dir, "B9C8_Ma")

flist_ga <- list.files(original_ga_dir, pattern = NULL, all.files = TRUE);
flist_na <- list.files(original_na_dir, pattern = NULL, all.files = TRUE);
flist_da <- list.files(original_da_dir, pattern = NULL, all.files = TRUE);
flist_ra <- list.files(original_ra_dir, pattern = NULL, all.files = TRUE);
flist_ma <- list.files(original_ma_dir, pattern = NULL, all.files = TRUE);

file.copy(file.path(original_ga_dir, flist_ga), file.path(my_ga_dir))
file.copy(file.path(original_na_dir, flist_na), file.path(my_na_dir))
file.copy(file.path(original_da_dir, flist_da), file.path(my_da_dir))
file.copy(file.path(original_ra_dir, flist_ra), file.path(my_ra_dir))
file.copy(file.path(original_ma_dir, flist_ma), file.path(my_ma_dir))

# change file name like "Ga_01.tif"
fname_change <- function(flist, base_dir, base_fname) {
	new_idx=1
	for(i in 3:length(flist)) {
		# set prev file name
		prev_fname <- paste("/", base_dir, flist[i], sep="/");
		# set new file name
		new_fname <- sprintf("%s%d.tif", base_fname, new_idx);
		new_fname <- paste("/", base_dir, new_fname, sep="/");
		# rename
		file.rename(from=prev_fname, to=new_fname);
		new_idx <- new_idx + 1;
	}
}

fname_change(flist_ga, my_ga_dir, "Ga_")
fname_change(flist_na, my_na_dir, "Na_")
fname_change(flist_da, my_da_dir, "Da_")
fname_change(flist_ra, my_ra_dir, "Ra_")
fname_change(flist_ma, my_ma_dir, "Ma_")

## mkdir train, validation, test
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

dir.create(train_dir)
dir.create(validation_dir)
dir.create(test_dir)

train_ga_dir <- file.path(train_dir, "Ga/")
train_na_dir <- file.path(train_dir, "Na/")
train_da_dir <- file.path(train_dir, "Da/")
train_ra_dir <- file.path(train_dir, "Ra/")
train_ma_dir <- file.path(train_dir, "Ma/")

validation_ga_dir <- file.path(validation_dir, "Ga/")
validation_na_dir <- file.path(validation_dir, "Na/")
validation_da_dir <- file.path(validation_dir, "Da/")
validation_ra_dir <- file.path(validation_dir, "Ra/")
validation_ma_dir <- file.path(validation_dir, "Ma/")

test_ga_dir <- file.path(test_dir, "Ga/")
test_na_dir <- file.path(test_dir, "Na/")
test_da_dir <- file.path(test_dir, "Da/")
test_ra_dir <- file.path(test_dir, "Ra/")
test_ma_dir <- file.path(test_dir, "Ma/")

dir.create(train_ga_dir)
dir.create(train_na_dir)
dir.create(train_da_dir)
dir.create(train_ra_dir)
dir.create(train_ma_dir)
dir.create(test_ga_dir)
dir.create(test_na_dir)
dir.create(test_da_dir)
dir.create(test_ra_dir)
dir.create(test_ma_dir)
dir.create(validation_ga_dir)
dir.create(validation_na_dir)
dir.create(validation_da_dir)
dir.create(validation_ra_dir)
dir.create(validation_ma_dir)

# copy images to train, validation, test
copy_files <- function(base_name, num, src_dir, dst_dir) {
	fnames <- paste0(base_name, num, ".tif")
	file.copy(file.path(src_dir, fnames),
			  file.path(dst_dir))
}

copy_files("Ga_", 1:600, my_ga_dir, train_ga_dir)
copy_files("Na_", 1:600, my_na_dir, train_na_dir)
copy_files("Da_", 1:600, my_da_dir, train_da_dir)
copy_files("Ra_", 1:600, my_ra_dir, train_ra_dir)
copy_files("Ma_", 1:600, my_ma_dir, train_ma_dir)

copy_files("Ga_", 601:720, my_ga_dir, validation_ga_dir)
copy_files("Na_", 601:720, my_na_dir, validation_na_dir)
copy_files("Da_", 601:720, my_da_dir, validation_da_dir)
copy_files("Ra_", 601:720, my_ra_dir, validation_ra_dir)
copy_files("Ma_", 601:720, my_ma_dir, validation_ma_dir)

copy_files("Ga_", 721:length(flist_ga)-2, my_ga_dir, test_ga_dir)
copy_files("Na_", 721:length(flist_na)-2, my_na_dir, test_na_dir)
copy_files("Da_", 721:length(flist_da)-2, my_da_dir, test_da_dir)
copy_files("Ra_", 721:length(flist_ra)-2, my_ra_dir, test_ra_dir)
copy_files("Ma_", 721:length(flist_ma)-2, my_ma_dir, test_ma_dir)

#################### CNN HanGul ####################

library("keras")
library("tensorflow")

## preprocess
train_datagen <- image_data_generator(rescale=1/255)
test_datagen <- image_data_generator(rescale=1/255)
validation_datagen <- image_data_generator(rescale=1/255)

train_generator <- flow_images_from_directory(
	train_dir,
	train_datagen,
	target_size = c(150, 150),
	batch_size = 20,
	class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
	validation_dir,
	validation_datagen,
	target_size = c(150, 150),
	batch_size = 20,
	class_mode = "categorical"
)

test_generator <- flow_images_from_directory(
	test_dir,
	test_datagen,
	target_size = c(150, 150),
	batch_size = 20,
	class_mode = "categorical"
)

##### CNN Neural Network
model <- keras_model_sequential() %>%
	layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu",                          input_shape=c(150, 150, 3)) %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_flatten() %>%
	layer_dense(units=512, activation="relu") %>%
	layer_dense(units=5, activation="softmax")

print(model)

##### compile CNN
model %>% compile(
	optimizer = optimizer_rmsprop(lr = 1e-3),
	loss="categorical_crossentropy",
	metrics=c("acc")
)


##### learning with fit_generator()
history <- model %>% fit_generator(
	train_generator,
	steps_per_epoch = 30,
	epochs = 25,
	validation_data = validation_generator,
	validation_steps = 6
)

plot(history)

##### evaluate
model %>% evaluate_generator(test_generator, steps=10)

##### save model
fname = "my6"
dir = "./CNN_Hangul_Data/my"
model %>% save_model_hdf5(paste(dir, fname, ".h5", sep=""))
png(paste(dir, fname, ".png", sep=""))
plot(history)
dev.off()


##### load model
model <- load_model_hdf5("./CNN_Hangul_Data/my/my6.h5")

##### test my hangul
image_file <- "./CNN_Hangul_Data/mine/my_Ma3.tif"
img <- image_load(image_file, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255
plot(as.raster(img_tensor[1,,,]))

##### predict
result <- model %>% predict(img_tensor)
print(result)

##### Visualization
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
activation_model <- keras_model(input=model$input, outputs=layer_outputs)
activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)

plot_channel <- function(channel) {
	rotate <- function(x) t(apply(x, 2, rev))
	image(rotate(channel), axes=FALSE, asp=1,
		col=terrain.colors(12))
}


plot_channel(first_layer_activation[1,,,1])
plot_channel(first_layer_activation[1,,,2])

# draw all channels and save as png files

image_size <- 58
images_per_row <- 16

for (i in 1:8) {
	layer_activation <- activations[[i]]
	layer_name <- model$layers[[i]]$name
	
	n_features <- dim(layer_activation)[[4]]
	ncols <- n_features %/% images_per_row
	
	png(paste0("actovations_", i, "_", layer_name, ".png"),
		width=image_size * images_per_row,
		height = image_size * ncols)
	op <- par(mfrow = c(ncols, images_per_row), mai=rep_len(0.02, 4))
	
	for (col in 0:(ncols-1)) {
		for (row in 0:(images_per_row-1)) {
			channel_image <- layer_activation[1,,,(col*images_per_row)+row+1]
			plot_channel(channel_image)
		}
	}
	par(op)
	dev.off()
}





# EOF
