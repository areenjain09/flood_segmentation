#!/usr/bin/env bash
set -u

err_file="$(mktemp)"
trap 'rm -f "$err_file"' EXIT

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"
cd "$project_dir" || exit 1

Rscript "$script_dir/train.R" "$@" 2> "$err_file"
status=$?

grep -v "NNPACK.cpp:56.*Could not initialize NNPACK" "$err_file" >&2 || true
exit "$status"
