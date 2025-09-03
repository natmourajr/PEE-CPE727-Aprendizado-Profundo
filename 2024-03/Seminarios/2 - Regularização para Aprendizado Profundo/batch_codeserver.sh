#!/bin/bash
#SBATCH --job-name=%u-code-server
#SBATCH --partition=cpu
#SBATCH --time=2-12:00:00
#SBATCH --mem=8GB
#SBATCH --output=/home/%u/code-server.log

singularity exec --nv -H $PWD:/home cpe727-regularization.sif bash -c "(code-server --bind-addr localhost:10100 &) && cat /home/.config/code-server/config.yaml"