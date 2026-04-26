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
source(file.path(project_dir, "lib", "model.R"))
source(file.path(project_dir, "lib", "visualize.R"))

resolve_path <- function(path) {
  if (grepl("^/", path)) {
    return(path)
  }
  normalizePath(file.path(project_dir, path), mustWork = FALSE)
}

args <- parse_cli_args(list(
  "image" = "",
  "checkpoint" = "outputs/checkpoints/best_model.pt",
  "image-size" = "128",
  "base-channels" = "8",
  "output" = "outputs/predictions/single_mask.png"
))

if (args[["image"]] == "") {
  stop("Please provide an image path with --image path/to/image.jpg")
}

image_path <- resolve_path(args[["image"]])
checkpoint_path <- resolve_path(args[["checkpoint"]])
output_path <- resolve_path(args[["output"]])

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
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

image_tensor <- read_image_tensor(image_path, as_int(args[["image-size"]]))$unsqueeze(1)$to(device = device)
logits <- model(image_tensor)
predicted_mask <- (torch_sigmoid(logits[1, , , ]) >= 0.5)$to(dtype = torch_float())

save_mask_png(predicted_mask, output_path)
cat(sprintf("Saved predicted mask to: %s\n", output_path))
