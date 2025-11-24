# Copyright (c) Meta Platforms, Inc. and affiliates.

DEST_DIR="$(pwd)/data"
mkdir -p "$DEST_DIR"

NPYS=(
    test_split_DW4.npy
    test_split_LJ13-1000.npy
    test_split_LJ55-1000-part1.npy
)

DEM_GITHUB="https://raw.githubusercontent.com/jarridrb/DEM/main/data"

for NPY in "${NPYS[@]}"; do
  curl -o "$DEST_DIR/$NPY" "$DEM_GITHUB/$NPY"
done

echo "Files downloaded to $DEST_DIR"
