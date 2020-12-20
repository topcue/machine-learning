## IMDB
#############################################

setwd("/Users/topcue/myR")
rm(list=ls())

# install.packages("keras")
# install.packages("tensorflow")
# tensorflow::install_tensorflow()

#################### IMDB ####################
library(keras)

##### get IMDB data
imdb <- dataset_imdb(num_words=10000)

###### data split
train_data <- imdb$train$x
train_labels <- imdb$train$y

test_data <- imdb$test$x
test_labels <- imdb$test$y

val_indices <- 1:10000
x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# 하나의 영화평 (각 숫자는 몇번째 단어인지를 나타냄)
print(train_data[[1]])

##### Decode review (int --> sentences)
## word_index:(word, int) --> reverse_word_index:(int, word)
word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

print(reverse_word_index[1])
print(reverse_word_index["34701"])

## decode train_data[[1]] 
# (vector of int) --> (sentences of words)
decoded_review_1 <- sapply(train_data[[1]], function(index) {
	word <- if (index >= 3) reverse_word_index[as.character(index-3)]
	if(!is.null(word)) word else "?"
})

print(paste(decoded_review_1, collapse=" "))

## decode specific train_data[[n]]
foo <-  function(index) {
	word <- if (index >= 3) reverse_word_index[as.character(index-3)]
	if(!is.null(word)) word else "?"
}

decode_review_num <- function(num) {
	return (sapply(train_data[[num]], foo))
}
print(decode_review_num(1))
print(paste(decode_review_num(1), collapse=" "))

## decode all train_data (remove "[1:3]")
decode_review_all <- lapply(train_data[1:3], function(vec){
	sapply(vec, foo)
})
print(decode_review_all)
print(paste(decode_review_all[[1]], collapse=" "))
print(paste(decode_review_all[[2]], collapse=" "))

##### vectorizing
vectorize_sequences <- function(sequences, dimension=10000) {
	# result [25000, 10000]
	results <- matrix(0, nrow=length(sequences), ncol=dimension)
	for (i in 1:length(sequences))
		# sequences[[i]] : i-th review
		results[i, sequences[[i]]] <- 1
	return (results)
}

x_train <- vectorize_sequences(train_data )
x_test <- vectorize_sequences(test_data)

# 첫번째 감상평 (1은 n번째 단어가 리뷰에 나왔다는 뜻)
print(x_train[1, ])

# convert int to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

##### Neural Network Model
model <- keras_model_sequential() %>%
	layer_dense(units=16, activation="relu", input_shape=c(10000)) %>%
	layer_dense(units=16, activation="relu") %>%
	layer_dense(units=1, activation="sigmoid")

print(model)

##### compile
model %>% compile(
	optimizer = "rmsprop",
	loss = "binary_crossentropy",
	metrics = c("accuracy")
)

##### validation set (part of training set)


##### learning
history <- model %>% fit(
	partial_x_train,
	partial_y_train,
	epoch=20,
	batch_size=512,
	validation_data=list(x_val, y_val)
)

print(history)
plot(history)

##### validation
model %>% predict(x_test[1:10,])
y_test[1:10]

##### evaluation
result <- metrics <- model %>% evaluate(x_test, y_test)
print(result)


# EOF
