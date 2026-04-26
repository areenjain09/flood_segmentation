library(torch)
library(magick)

image_extensions <- c("jpg", "jpeg", "png", "bmp", "tif", "tiff")

normalize_stem <- function(path) {
  stem <- tools::file_path_sans_ext(basename(path))
  stem <- tolower(stem)
  tokens <- c("_mask", "-mask", " mask", "_label", "-label", " label")
  for (token in tokens) {
    stem <- gsub(token, "", stem, fixed = TRUE)
  }
  stem
}

path_has_part <- function(path, names) {
  parts <- unlist(strsplit(normalizePath(path, winslash = "/", mustWork = FALSE), "/", fixed = TRUE))
  any(tolower(parts) %in% names)
}

discover_pairs <- function(data_dir) {
  if (!dir.exists(data_dir)) {
    stop(sprintf("Data directory does not exist: %s", data_dir))
  }

  files <- list.files(data_dir, recursive = TRUE, full.names = TRUE)
  exts <- tolower(tools::file_ext(files))
  image_files <- files[exts %in% image_extensions]

  mask_files <- image_files[vapply(
    image_files,
    path_has_part,
    logical(1),
    names = c("mask", "masks", "label", "labels")
  )]

  image_files <- setdiff(image_files, mask_files)
  image_dir_files <- image_files[vapply(
    image_files,
    path_has_part,
    logical(1),
    names = c("image", "images", "img", "imgs")
  )]

  if (length(image_dir_files) > 0) {
    image_files <- image_dir_files
  } else {
    image_files <- image_files[!grepl("mask|label", basename(image_files), ignore.case = TRUE)]
  }

  mask_stems <- vapply(mask_files, normalize_stem, character(1))
  names(mask_files) <- mask_stems

  pairs <- lapply(image_files, function(image_path) {
    stem <- normalize_stem(image_path)
    mask_path <- unname(mask_files[[stem]])
    if (is.null(mask_path) || is.na(mask_path)) {
      return(NULL)
    }
    data.frame(image_path = image_path, mask_path = mask_path, stringsAsFactors = FALSE)
  })

  pairs <- do.call(rbind, pairs[!vapply(pairs, is.null, logical(1))])
  if (is.null(pairs) || nrow(pairs) == 0) {
    stop("No image-mask pairs found. Expected folders such as images/ and masks/.")
  }

  pairs[order(basename(pairs$image_path)), , drop = FALSE]
}

load_test_filenames <- function(path) {
  if (is.null(path) || path == "" || !file.exists(path)) {
    return(NULL)
  }

  csv <- read.csv(path, stringsAsFactors = FALSE)
  if (!"filename" %in% names(csv)) {
    stop(sprintf("Test filename CSV must contain a filename column: %s", path))
  }
  unique(csv$filename)
}

split_pairs <- function(pairs, val_size = 0.2, test_size = 0.1, seed = 42, test_filenames = NULL) {
  if (nrow(pairs) < 3) {
    stop("Need at least 3 image-mask pairs to create train/val/test splits.")
  }

  set.seed(seed)
  if (!is.null(test_filenames)) {
    is_test <- basename(pairs$image_path) %in% test_filenames
    test <- pairs[is_test, , drop = FALSE]
    train_val <- pairs[!is_test, , drop = FALSE]
    if (nrow(test) == 0) {
      stop("No R test rows matched the provided Python test filenames.")
    }
    test_order <- match(test_filenames, basename(test$image_path))
    test <- test[test_order[!is.na(test_order)], , drop = FALSE]
  } else {
    test_count <- max(1, round(nrow(pairs) * test_size))
    test_idx <- sample(seq_len(nrow(pairs)), test_count)
    train_val <- pairs[-test_idx, , drop = FALSE]
    test <- pairs[test_idx, , drop = FALSE]
  }

  val_count <- max(1, round(nrow(pairs) * val_size))
  val_idx <- sample(seq_len(nrow(train_val)), val_count)
  val <- train_val[val_idx, , drop = FALSE]
  train <- train_val[-val_idx, , drop = FALSE]

  list(train = train, val = val, test = test)
}

read_image_tensor <- function(path, image_size) {
  image <- image_read(path)
  image <- image_convert(image, colorspace = "sRGB")
  image <- image_resize(image, sprintf("%dx%d!", image_size, image_size))
  array <- image_data(image, channels = "rgb")
  values <- aperm(as.integer(array) / 255, c(3, 2, 1))
  torch_tensor(values)$to(dtype = torch_float())
}

read_mask_tensor <- function(path, image_size) {
  mask <- image_read(path)
  mask <- image_convert(mask, colorspace = "gray")
  mask <- image_resize(mask, sprintf("%dx%d!", image_size, image_size), filter = "point")
  array <- image_data(mask, channels = "gray")
  values <- aperm(as.integer(array) > 127, c(3, 2, 1))
  torch_tensor(values)$to(dtype = torch_float())
}

flood_dataset <- dataset(
  name = "FloodDataset",

  initialize = function(pairs, image_size = 256) {
    self$pairs <- pairs
    self$image_size <- image_size
  },

  .getitem = function(index) {
    image_path <- self$pairs$image_path[[index]]
    mask_path <- self$pairs$mask_path[[index]]

    list(
      x = read_image_tensor(image_path, self$image_size),
      y = read_mask_tensor(mask_path, self$image_size),
      filename = basename(image_path)
    )
  },

  .length = function() {
    nrow(self$pairs)
  }
)
