## CNN MNIST
#############################################

setwd("/Users/topcue/myR")
rm(list=ls())

# install.packages("keras")
# install.packages("tensorflow")
# tensorflow::install_tensorflow()

#################### CNN MNISTl ####################
rm(list=ls())

library("keras")
library("tensorflow")

##### load data
mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

##### reshape and normalize
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images/255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images/255

##### categoricalize labels
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

##### CNN Neural Network
cnn_model <- keras_model_sequential() %>%
	layer_conv_2d(filters=32, kernel_size=c(3,3), activation="relu",                          input_shape=c(28, 28, 1)) %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
	layer_max_pooling_2d(pool_size=c(2, 2)) %>%
	layer_conv_2d(filters=64, kernel_size=c(3,3), activation="relu") %>%
	layer_flatten() %>%
	layer_dense(units=64, activation="relu") %>%
	layer_dense(units=10, activation="softmax")

print(cnn_model)

##### compile
cnn_model %>% compile(
	optimizer="rmsprop",
	loss="categorical_crossentropy",
	metrics=c("accuracy")
)

##### learning
cnn_model %>% fit(
	train_images, train_labels,
	epochs=10,
	batch_size=64
)

##### evaluation
results <- cnn_model %>% evaluate(test_images, test_labels)
print(results)

# EOF
