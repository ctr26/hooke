#!/bin/bash

FEATURES_DIR="/mnt/ps/home/CORP/jason.hartford/project/big-x/metrics/hooke-mini/step_150000/features"
METADATA_PATH="/mnt/ps/home/CORP/jason.hartford/project/big-x/big-img/metadata"

mkdir -p "$FEATURES_DIR/dart"
mkdir -p "$FEATURES_DIR/cell_diversity"

ln -s "$METADATA_PATH/dart_inference_set.parquet" "$FEATURES_DIR/dart/obs.parquet"
ln -s "$METADATA_PATH/celltype_diversity_inference_set.parquet" "$FEATURES_DIR/cell_diversity/obs.parquet"

ln -s "$FEATURES_DIR/pred_phenom.zarr" "$FEATURES_DIR/dart/features.zarr"
ln -s "$FEATURES_DIR/pred_phenom.zarr" "$FEATURES_DIR/cell_diversity/features.zarr"
