## Convolution Filter
#############################################

setwd("/Users/topcue/myR")
rm(list=ls())

# install.packages("magick")

############## Convolution Filter ############
library(magick)

image_ogu <- image_read("./Convolution_Filter_Data/ogu.jpeg")
print(image_ogu)

# edge filter
edge_filter <- matrix(c(0, 1, 0, -1, -4, -1, 0, -1, 4), 3, 3)
print(edge_filter)
image_ogu_edge <- image_convolve(image_ogu, edge_filter)
print(image_ogu_edge)

# edge filter 2
edge_filter2 <- matrix(c(-1, -1, -1, -1, 8, -1, -1, -1, -1), 3, 3)
image_ogu_edge2 <- image_convolve(image_ogu, edge_filter2)
print(image_ogu_edge2)

# edge filter 3
edge_filter2 <- matrix(c(1, 1, 1, 1, -8, 1, 1, 1, 1), 3, 3)
image_ogu_edge3 <- image_convolve(image_ogu, edge_filter2)
print(image_ogu_edge3)

# sharpen filter
sharpen_filter <- matrix(c(0, -1, 0, -1, 6, -1, 0, -1, 0), 3, 3)
image_ogu_sharpen <- image_convolve(image_ogu, sharpen_filter)
print(image_ogu_sharpen)

# remove horizen filter
sobelH_filter <- matrix(c(1, 2, 1, 0, 0, 0, -1, -2, -1), 3, 3)
image_ogu_sobelH <- image_convolve(image_ogu, sobelH_filter)
print(image_ogu_sobelH)

# removing vertical filter
sobelV_filter <- t(sobelH_filter)
image_ogu_sobelV <- image_convolve(image_ogu, sobelV_filter)
print(image_ogu_sobelV)

# EOF
