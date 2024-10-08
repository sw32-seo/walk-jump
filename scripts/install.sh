#!/usr/bin/env bash

set -eux

if command -v mamba
then
 mamba env create --file env.yaml
else
 conda env create --file env.yaml
fi

eval "$(conda shell.bash hook)"
conda activate wj
pip install -e .

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
