parse_cli_args <- function(defaults = list()) {
  raw_args <- commandArgs(trailingOnly = TRUE)
  args <- defaults

  if (length(raw_args) == 0) {
    return(args)
  }

  idx <- 1
  while (idx <= length(raw_args)) {
    key <- raw_args[[idx]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key))
    }

    name <- sub("^--", "", key)
    if (idx == length(raw_args) || startsWith(raw_args[[idx + 1]], "--")) {
      args[[name]] <- TRUE
      idx <- idx + 1
    } else {
      args[[name]] <- raw_args[[idx + 1]]
      idx <- idx + 2
    }
  }

  args
}

as_int <- function(value) {
  as.integer(value)
}

as_num <- function(value) {
  as.numeric(value)
}
