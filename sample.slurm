#!/bin/bash
#SBATCH --job-name=in5550
#SBATCH --account=ec403
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#
# By default, request four CPU cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on GPUs
#
#SBATCH --cpus-per-task=4

# NB: this script should be run with "sbatch sample.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
python3 sanity.py ${@}
