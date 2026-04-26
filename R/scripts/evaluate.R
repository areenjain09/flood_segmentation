get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))))
  }
  getwd()
}

project_dir <- normalizePath(file.path(get_script_dir(), ".."))
local_lib <- file.path(project_dir, "packages")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
torch_home <- file.path(project_dir, "torch-cache")
dir.create(torch_home, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(TORCH_HOME = torch_home)
Sys.setenv(TORCH_CPP_LOG_LEVEL = "ERROR")
Sys.setenv(C10_LOG_LEVEL = "ERROR")
Sys.setenv(GLOG_minloglevel = "2")
Sys.setenv(PYTORCH_DISABLE_NNPACK = "1")
source(file.path(project_dir, "lib", "args.R"))
source(file.path(project_dir, "lib", "data.R"))
source(file.path(project_dir, "lib", "metrics.R"))
source(file.path(project_dir, "lib", "model.R"))
source(file.path(project_dir, "lib", "visualize.R"))

resolve_path <- function(path) {
  if (grepl("^/", path)) {
    return(path)
  }
  normalizePath(file.path(project_dir, path), mustWork = FALSE)
}

args <- parse_cli_args(list(
  "data-dir" = "../python/data/raw",
  "checkpoint" = "outputs/checkpoints/best_model.pt",
  "image-size" = "128",
  "batch-size" = "1",
  "base-channels" = "8",
  "test-filenames" = "../python/outputs/test_predictions.csv",
  "output-csv" = "outputs/test_predictions.csv",
  "prediction-dir" = "outputs/predictions",
  "max-panels" = "10",
  "seed" = "42"
))

data_dir <- resolve_path(args[["data-dir"]])
checkpoint_path <- resolve_path(args[["checkpoint"]])
output_csv <- resolve_path(args[["output-csv"]])
prediction_dir <- resolve_path(args[["prediction-dir"]])
dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
dir.create(prediction_dir, recursive = TRUE, showWarnings = FALSE)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
pairs <- discover_pairs(data_dir)
test_filenames <- load_test_filenames(resolve_path(args[["test-filenames"]]))
splits <- split_pairs(pairs, seed = as_int(args[["seed"]]), test_filenames = test_filenames)
if (!is.null(test_filenames)) {
  cat(sprintf("Using Python test filename split from: %s\n", resolve_path(args[["test-filenames"]])))
}

test_loader <- dataloader(
  flood_dataset(splits$test, image_size = as_int(args[["image-size"]])),
  batch_size = as_int(args[["batch-size"]]),
  shuffle = FALSE
)

checkpoint <- torch_load(checkpoint_path)
checkpoint_args <- checkpoint$args
base_channels <- if (!is.null(checkpoint_args) && !is.null(checkpoint_args[["base-channels"]])) {
  as_int(checkpoint_args[["base-channels"]])
} else {
  as_int(args[["base-channels"]])
}
model <- unet_model(base_channels = base_channels)$to(device = device)
model$load_state_dict(checkpoint$model_state_dict)
model$eval()

results <- data.frame(filename = character(), dice = numeric(), iou = numeric())
saved_panels <- 0

coro::loop(for (batch in test_loader) {
  images <- batch$x$to(device = device)
  masks <- batch$y$to(device = device)
  logits <- model(images)
  batch_size <- images$size(1)

  for (i in seq_len(batch_size)) {
    item_logits <- logits[i, , , , drop = FALSE]
    item_mask <- masks[i, , , , drop = FALSE]
    item_dice <- dice_score(item_logits, item_mask)$item()
    item_iou <- iou_score(item_logits, item_mask)$item()
    filename <- batch$filename[[i]]

    results <- rbind(
      results,
      data.frame(filename = filename, dice = item_dice, iou = item_iou)
    )

    if (saved_panels < as_int(args[["max-panels"]])) {
      save_prediction_panel(
        images[i, , , ],
        masks[i, , , ],
        logits[i, , , ],
        file.path(prediction_dir, sprintf("%s_prediction.png", tools::file_path_sans_ext(filename)))
      )
      saved_panels <- saved_panels + 1
    }
  }
})

write.csv(results, output_csv, row.names = FALSE)
cat(sprintf("Test Dice: %.4f\n", mean(results$dice)))
cat(sprintf("Test IoU: %.4f\n", mean(results$iou)))
cat(sprintf("Saved per-image results to: %s\n", output_csv))
cat(sprintf("Saved prediction panels to: %s\n", prediction_dir))
