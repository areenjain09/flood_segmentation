args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
project_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
} else {
  getwd()
}

local_lib <- file.path(project_dir, "packages")
dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(local_lib, .libPaths()))

torch_home <- file.path(project_dir, "torch-cache")
dir.create(torch_home, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(TORCH_HOME = torch_home)
options(pkgType = "binary")

packages <- c("torch", "magick", "png")

missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, lib = local_lib, repos = "https://cloud.r-project.org", type = "binary")
}

if (requireNamespace("torch", quietly = TRUE)) {
  torch::install_torch()
}

message("R package setup complete.")
