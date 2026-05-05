#!/bin/bash
# Path to your conda initialization script
source /home/ken/anaconda3/etc/profile.d/conda.sh
# Activate the desired conda environment
conda activate stock
cd /home/ken/git/NEAT
# Run the batch processing script
python3 claude-2-mul-thread-gpt5-vol-selection-output-ckpoint.py --max-population=2000 \
--log-file=data_vol-cluster-0-startdate-1962-01-03.log  \
--input-file=data_vol-cluster-0-startdate-1962-01-03.csv  \
--output-file=data_vol-cluster-0-startdate-1962-01-03.out.csv \
--json-output-dir=./json-output \
--checkpoint-dir=checkpoints-startdate-1962-01-03 \
--reload --long-only

python3 claude-4-new-NEAT-log-species-v2-process-opti-fix.py --max-population=8000 --log-file=data_vol-cluster-0-startdate-1962-01-03-species-process-opti-fix.log  --input-file=data_vol-cluster-0-startdate-1962-01-03.csv  --output-file=data_vol-cluster-0-startdate-1962-01-03-species-process-opti-fix.out.csv --json-output-dir=./json-output-species-v2-process-opti-fix --checkpoint-dir=checkpoints-startdate-1962-01-03-species-v2-process-opti-fix  --long-only


cp -rf ./json-output/* ../orch/models_output_json/NEAT_62/
cp -rf ./json-output-species-v2-process-opti-fix/* ../orch/models_output_json/NEAT_62_species-v2-process-opti-fix
