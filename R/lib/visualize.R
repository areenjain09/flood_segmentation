library(torch)
library(png)

tensor_image_array <- function(image_tensor) {
  arr <- as.array(image_tensor$detach()$cpu())
  if (length(dim(arr)) == 3 && dim(arr)[1] == 3) {
    arr <- aperm(arr, c(2, 3, 1))
  }
  pmin(pmax(arr, 0), 1)
}

tensor_mask_matrix <- function(mask_tensor) {
  arr <- as.array(mask_tensor$detach()$cpu())
  if (length(dim(arr)) == 3) {
    arr <- arr[1, , ]
  }
  arr
}

draw_mask <- function(mask_matrix, title) {
  image(
    t(mask_matrix[nrow(mask_matrix):1, ]),
    col = c("black", "white"),
    axes = FALSE,
    main = title
  )
}

save_prediction_panel <- function(image, mask, logits, output_path, threshold = 0.5) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  image_arr <- tensor_image_array(image)
  mask_mat <- tensor_mask_matrix(mask)
  pred_mat <- tensor_mask_matrix((torch_sigmoid(logits) >= threshold)$to(dtype = torch_float()))

  png(output_path, width = 1200, height = 400)
  old_par <- par(mfrow = c(1, 3), mar = c(1, 1, 3, 1))
  on.exit({
    par(old_par)
    dev.off()
  })

  plot.new()
  rasterImage(image_arr, 0, 0, 1, 1)
  title("Image")

  draw_mask(mask_mat, "Ground Truth")
  draw_mask(pred_mat, "Prediction")
}

save_mask_png <- function(mask_tensor, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  mask_mat <- tensor_mask_matrix(mask_tensor)
  writePNG(mask_mat, output_path)
}
