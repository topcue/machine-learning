## Set Up Directory
## CNN Cat Dog
#############################################

rm(list=ls())
setwd("/Users/topcue/CNN_Cat_Dog_Data")

# install.packages("keras")
# install.packages("tensorflow")
# tensorflow::install_tensorflow()

# rstudioapi::restartSession()
# reticulate::use_python('/Users/topcue/Library/r-miniconda/envs/r-reticulate/bin/python', required = TRUE)

# reticulate::py_config()
# reticulate::py_install('pillow', pip=TRUE)
# reticulate::py_install('spicy', pip=TRUE)

############## Set Up Directory##############

## unzip Kaggle_Image.zip

## Data folder for training
base_dir <- "/Users/topcue/myR/CNN_Cat_Dog_Data"
original_dataset_dir <- "/Users/topcue/myR/CNN_Cat_Dog_Data/Kaggle_Image"

## mkdir train, validation, test
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

dir.create(train_dir)
dir.create(validation_dir)
dir.create(test_dir)

## mkdir cats, dogs
train_cats_dir <- file.path(train_dir, "cats")
train_dogs_dir <- file.path(train_dir, "dogs")
validation_cats_dir <- file.path(validation_dir, "cats")
validation_dogs_dir <- file.path(validation_dir, "dogs")
test_cats_dir <- file.path(test_dir, "cats")
test_dogs_dir <- file.path(test_dir, "dogs")

dir.create(validation_cats_dir)
dir.create(train_cats_dir)
dir.create(train_dogs_dir)
dir.create(validation_dogs_dir)
dir.create(test_cats_dir)
dir.create(test_dogs_dir)

## copy image files
fnames <- paste0("cat/", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
		  file.path(train_cats_dir))
fnames <- paste0("cat/", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
		  file.path(validation_cats_dir))
fnames <- paste0("cat/", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
		  file.path(test_cats_dir))
fnames <- paste0("dog/", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
		  file.path(train_dogs_dir))
fnames <- paste0("dog/", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
		  file.path(validation_dogs_dir))
fnames <- paste0("dog/", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames),
		  file.path(test_dogs_dir))

## copy image files
cat("The number of training cat images: ",
	length(list.files(train_cats_dir)), "\n")
cat("The number of training dog images: ",
	length(list.files(train_dogs_dir)), "\n")
cat("The number of validation cat images: ",
	length(list.files(validation_cats_dir)), "\n")
cat("The number of validation cat images: ",
	length(list.files(validation_dogs_dir)), "\n")
cat("The number of test cat images: ",
	length(list.files(test_cats_dir)), "\n")
cat("The number of test cat images: ",
	length(list.files(test_dogs_dir)), "\n")

#################### CNN Cat Dog ####################

library("keras")
library("tensorflow")

##### preprocess
train_datagen <- image_data_generator(rescale=1/255)
validation_datagen <- image_data_generator(rescale=1/255)

train_generator <- flow_images_from_directory(
	train_dir,
	train_datagen,
	target_size = c(150, 150),
	batch_size = 20,
	class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
	validation_dir,
	validation_datagen,
	target_size = c(150, 150),
	batch_size = 20,
	class_mode = "binary"
)

batch <- generator_next(train_generator)
str(batch)

batch <- generator_next(train_generator)
str(batch)

##### CNN Neural Network
cnn_model <- keras_model_sequential() %>%
	layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu",                          input_shape=c(150, 150, 3)) %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=128, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_flatten() %>%
	layer_dense(units=256, activation="relu") %>%
	layer_dense(units=1, activation="sigmoid")

print(cnn_model)

##### compile CNN
cnn_model %>% compile(
	optimizer = optimizer_rmsprop(lr = 1e-3), # learning rate=0.0001
	loss="binary_crossentropy",
	metrics=c("acc")
)

##### learning
history <- cnn_model %>% fit_generator(
	train_generator,
	steps_per_epoch = 30,
	epochs = 30,
	validation_data = validation_generator,
	validation_steps = 6
)

plot(history)

##### save model
cnn_model %>% save_model_hdf5("./CNN_Cat_Dog_Data/cat_dog.h5")

##### predict w/ ogu
image_file <- "./CNN_Cat_Dog_Data/ogu.jpeg"
img <- image_load(image_file, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor/255
plot(as.raster(img_tensor[1,,,]))

# cat = 1
result <- cnn_model %>% evaluate(img_tensor, 1)
print(result)

# EOF
