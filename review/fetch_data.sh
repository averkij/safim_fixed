#!/bin/bash
# Download the SAFIM dataset files used by the scripts in this directory.
cd "$(dirname "$0")" && mkdir -p data && cd data
for f in api_completion block_completion block_completion_v2 control_completion control_completion_fixed; do
  curl -sSL -O "https://huggingface.co/datasets/gonglinyuan/safim/resolve/main/$f.jsonl.gz"
done
