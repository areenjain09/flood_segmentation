library(torch)

conv_block <- nn_module(
  name = "ConvBlock",

  initialize = function(in_channels, out_channels) {
    self$block <- nn_sequential(
      nn_conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
      nn_batch_norm2d(out_channels),
      nn_relu(inplace = TRUE),
      nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
      nn_batch_norm2d(out_channels),
      nn_relu(inplace = TRUE)
    )
  },

  forward = function(x) {
    self$block(x)
  }
)

unet_model <- nn_module(
  name = "FloodUNet",

  initialize = function(base_channels = 8) {
    c1 <- base_channels
    c2 <- base_channels * 2
    c3 <- base_channels * 4
    c4 <- base_channels * 8

    self$enc1 <- conv_block(3, c1)
    self$pool1 <- nn_max_pool2d(kernel_size = 2)
    self$enc2 <- conv_block(c1, c2)
    self$pool2 <- nn_max_pool2d(kernel_size = 2)
    self$enc3 <- conv_block(c2, c3)
    self$pool3 <- nn_max_pool2d(kernel_size = 2)

    self$bottleneck <- conv_block(c3, c4)

    self$up3 <- nn_conv_transpose2d(c4, c3, kernel_size = 2, stride = 2)
    self$dec3 <- conv_block(c3 * 2, c3)
    self$up2 <- nn_conv_transpose2d(c3, c2, kernel_size = 2, stride = 2)
    self$dec2 <- conv_block(c2 * 2, c2)
    self$up1 <- nn_conv_transpose2d(c2, c1, kernel_size = 2, stride = 2)
    self$dec1 <- conv_block(c1 * 2, c1)

    self$out <- nn_conv2d(c1, 1, kernel_size = 1)
  },

  forward = function(x) {
    enc1 <- self$enc1(x)
    enc2 <- self$enc2(self$pool1(enc1))
    enc3 <- self$enc3(self$pool2(enc2))

    bottleneck <- self$bottleneck(self$pool3(enc3))

    dec3 <- self$up3(bottleneck)
    dec3 <- torch_cat(list(dec3, enc3), dim = 2)
    dec3 <- self$dec3(dec3)

    dec2 <- self$up2(dec3)
    dec2 <- torch_cat(list(dec2, enc2), dim = 2)
    dec2 <- self$dec2(dec2)

    dec1 <- self$up1(dec2)
    dec1 <- torch_cat(list(dec1, enc1), dim = 2)
    dec1 <- self$dec1(dec1)

    self$out(dec1)
  }
)
