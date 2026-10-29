#!/usr/bin/env bash
# Download the open datasets the engine uses (github.com/datasets).
set -euo pipefail
cd "$(dirname "$0")/data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/data/raw"; cd "$(dirname "$0")/data/raw"; }
base=https://raw.githubusercontent.com/datasets
curl -fsSL -o sp500_shiller.csv "$base/s-and-p-500/main/data/data.csv"
curl -fsSL -o cpi.csv           "$base/cpi-us/main/data/cpiai.csv"
curl -fsSL -o us10y_m.csv       "$base/bond-yields-us-10y/main/data/monthly.csv"
curl -fsSL -o gold_m.csv        "$base/gold-prices/main/data/monthly.csv"
curl -fsSL -o vix.csv           "$base/finance-vix/main/data/vix-daily.csv"
echo "downloaded to $(pwd)"
