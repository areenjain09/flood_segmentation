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

resolve_path <- function(path) {
  if (grepl("^/", path)) {
    return(path)
  }
  normalizePath(file.path(project_dir, path), mustWork = FALSE)
}

evaluate_model <- function(model, loader, device, criterion) {
  model$eval()
  total_loss <- 0
  total_dice <- 0
  total_iou <- 0
  batches <- 0

  coro::loop(for (batch in loader) {
    images <- batch$x$to(device = device)
    masks <- batch$y$to(device = device)
    logits <- model(images)

    total_loss <- total_loss + criterion(logits, masks)$item()
    total_dice <- total_dice + dice_score(logits, masks)$item()
    total_iou <- total_iou + iou_score(logits, masks)$item()
    batches <- batches + 1
  })

  list(
    loss = total_loss / max(batches, 1),
    dice = total_dice / max(batches, 1),
    iou = total_iou / max(batches, 1)
  )
}

args <- parse_cli_args(list(
  "data-dir" = "../python/data/raw",
  "epochs" = "10",
  "batch-size" = "1",
  "image-size" = "128",
  "base-channels" = "8",
  "lr" = "0.0001",
  "test-filenames" = "../python/outputs/test_predictions.csv",
  "checkpoint-dir" = "outputs/checkpoints",
  "metrics-csv" = "outputs/metrics.csv",
  "seed" = "42"
))

set.seed(as_int(args[["seed"]]))
torch_manual_seed(as_int(args[["seed"]]))

data_dir <- resolve_path(args[["data-dir"]])
checkpoint_dir <- resolve_path(args[["checkpoint-dir"]])
metrics_csv <- resolve_path(args[["metrics-csv"]])
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(metrics_csv), recursive = TRUE, showWarnings = FALSE)

pairs <- discover_pairs(data_dir)
test_filenames <- load_test_filenames(resolve_path(args[["test-filenames"]]))
splits <- split_pairs(pairs, seed = as_int(args[["seed"]]), test_filenames = test_filenames)
cat(sprintf(
  "Found %d pairs: %d train, %d val, %d test\n",
  nrow(pairs), nrow(splits$train), nrow(splits$val), nrow(splits$test)
))
if (!is.null(test_filenames)) {
  cat(sprintf("Using Python test filename split from: %s\n", resolve_path(args[["test-filenames"]])))
}

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
cat(sprintf("Using device: %s\n", device$type))

train_loader <- dataloader(
  flood_dataset(splits$train, image_size = as_int(args[["image-size"]])),
  batch_size = as_int(args[["batch-size"]]),
  shuffle = TRUE
)
val_loader <- dataloader(
  flood_dataset(splits$val, image_size = as_int(args[["image-size"]])),
  batch_size = as_int(args[["batch-size"]]),
  shuffle = FALSE
)

model <- unet_model(base_channels = as_int(args[["base-channels"]]))$to(device = device)
criterion <- dice_bce_loss()
optimizer <- optim_adamw(model$parameters, lr = as_num(args[["lr"]]))

metrics <- data.frame(
  epoch = integer(),
  train_loss = numeric(),
  val_loss = numeric(),
  val_dice = numeric(),
  val_iou = numeric()
)

best_iou <- -1

for (epoch in seq_len(as_int(args[["epochs"]]))) {
  model$train()
  total_train_loss <- 0
  train_batches <- 0
  expected_batches <- ceiling(nrow(splits$train) / as_int(args[["batch-size"]]))

  coro::loop(for (batch in train_loader) {
    images <- batch$x$to(device = device)
    masks <- batch$y$to(device = device)

    optimizer$zero_grad()
    logits <- model(images)
    loss <- criterion(logits, masks)
    loss$backward()
    optimizer$step()

    total_train_loss <- total_train_loss + loss$item()
    train_batches <- train_batches + 1

    if (train_batches %% 5 == 0 || train_batches == expected_batches) {
      cat(sprintf(
        "Epoch %d/%d batch %d/%d: train_loss=%.4f\n",
        epoch,
        as_int(args[["epochs"]]),
        train_batches,
        expected_batches,
        total_train_loss / train_batches
      ))
      flush.console()
    }
  })

  cat(sprintf("Epoch %d: running validation...\n", epoch))
  flush.console()
  val_metrics <- evaluate_model(model, val_loader, device, criterion)
  row <- data.frame(
    epoch = epoch,
    train_loss = total_train_loss / max(train_batches, 1),
    val_loss = val_metrics$loss,
    val_dice = val_metrics$dice,
    val_iou = val_metrics$iou
  )
  metrics <- rbind(metrics, row)
  write.csv(metrics, metrics_csv, row.names = FALSE)

  cat(sprintf(
    "Epoch %d: train_loss=%.4f, val_dice=%.4f, val_iou=%.4f\n",
    epoch, row$train_loss, row$val_dice, row$val_iou
  ))

  if (row$val_iou > best_iou) {
    best_iou <- row$val_iou
    torch_save(
      list(
        model_state_dict = model$state_dict(),
        args = args,
        val_iou = best_iou
      ),
      file.path(checkpoint_dir, "best_model.pt")
    )
  }
}

cat(sprintf("Best validation IoU: %.4f\n", best_iou))
