library(torch)

dice_score <- function(logits, masks, threshold = 0.5, eps = 1e-7) {
  preds <- (torch_sigmoid(logits) >= threshold)$to(dtype = torch_float())
  masks <- (masks >= 0.5)$to(dtype = torch_float())

  preds <- preds$flatten(start_dim = 2)
  masks <- masks$flatten(start_dim = 2)
  intersection <- (preds * masks)$sum(dim = 2)
  denominator <- preds$sum(dim = 2) + masks$sum(dim = 2)
  ((2 * intersection + eps) / (denominator + eps))$mean()
}

iou_score <- function(logits, masks, threshold = 0.5, eps = 1e-7) {
  preds <- (torch_sigmoid(logits) >= threshold)$to(dtype = torch_float())
  masks <- (masks >= 0.5)$to(dtype = torch_float())

  preds <- preds$flatten(start_dim = 2)
  masks <- masks$flatten(start_dim = 2)
  intersection <- (preds * masks)$sum(dim = 2)
  union <- preds$sum(dim = 2) + masks$sum(dim = 2) - intersection
  ((intersection + eps) / (union + eps))$mean()
}

dice_bce_loss <- nn_module(
  name = "DiceBCELoss",

  initialize = function() {
    self$bce <- nn_bce_with_logits_loss()
  },

  forward = function(logits, masks) {
    probs <- torch_sigmoid(logits)
    masks <- (masks >= 0.5)$to(dtype = torch_float())

    flat_probs <- probs$flatten(start_dim = 2)
    flat_masks <- masks$flatten(start_dim = 2)
    intersection <- (flat_probs * flat_masks)$sum(dim = 2)
    dice <- (2 * intersection + 1e-7) / (flat_probs$sum(dim = 2) + flat_masks$sum(dim = 2) + 1e-7)
    dice_loss <- 1 - dice$mean()

    self$bce(logits, masks) + dice_loss
  }
)
