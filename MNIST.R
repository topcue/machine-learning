## keras
###############################################

setwd("/Users/topcue/myR")
rm(list=ls())

# install.packages("keras")
# install.packages("tensorflow")
# tensorflow::install_tensorflow()

#################### keras ####################
library("keras")
library("tensorflow")

##### load data
mnist <- dataset_mnist()

##### train data set
train_images <- mnist$train$x
train_labels <- mnist$train$y

##### test data set
test_images <- mnist$test$x
test_labels <- mnist$test$y

train_images[1, , ]	# x1
train_labels[1]     # y2

train_images[2, , ] # x1
train_labels[2]     # y2

##### as.raster() : show image
plot(as.raster(train_images[1,,], max=255))

##### image reshaping and normalize
train_images <- array_reshape(train_images, c(60000, 28*28))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

##### categoricalize label
# for one-hot encoding
# (ex: 1 -> [0,1,0,0,0, ... , 0])
# (ex: 3 -> [0,0,0,3,0, ... , 0])
train_labels <- to_categorical(train_labels)

test_labels <- to_categorical(test_labels)

##### neural network
network <- keras_model_sequential() %>%
	layer_dense(units=512, activation="relu", input_shape=c(28*28)) %>%
	layer_dense(units=10, activation="softmax")

print(network)

##### compile
network %>% compile(
	optimizer = "rmsprop",
	loss = "categorical_crossentropy",
	metrics = c("accuracy")
)

##### fit
history <- network %>% fit(
	train_images,
	train_labels,
	epochs=5,
	batch_size=128
)

plot(history)

##### evaluate
metrics <- network %>% evaluate(test_images, test_labels)
print(metrics)

##### predict
network %>% predict(test_images[1:3,])
print(test_labels[1:3, ])


# EOF
