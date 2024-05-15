#!/bin/bash -x
#SBATCH --account=<your project>
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=inference_destine-model-out.%j
#SBATCH --error=inference_destine-model-err.%j
#SBATCH --time=02:00:00
##SBATCH --gres=gpu:1
##SBATCH --constraint=largedata
#SBATCH --partition=dc-cpu-devel
##SBATCH --partition=dc-gpu-devel
##SBATCH --partition=dc-cpu
##SBATCH --partition=dc-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_email>@fz-juelich.de

######### Template identifier (remove after customizing placeholders <...>) #########
echo "Do not run the template scripts"
exit 99
######### Template identifier (remove after customizing placeholders <...>) #########

# basic directories
WORK_DIR=$(pwd)
BASE_DIR=$(dirname "${WORK_DIR}")

# Name of virtual environment
VENV_DIR=${BASE_DIR}/virtual_envs/
VIRT_ENV_NAME=<your_venv>            # should be created with 'source create_env.sh <some_name>'

# Loading mouldes
source ../env_setup/modules_jsc.sh
# Activate virtual environment if needed (and possible)
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate ]]; then
      echo "Activating virtual environment..."
      source ${VENV_DIR}/${VIRT_ENV_NAME}/bin/activate
   else
      echo "ERROR: Requested virtual environment ${VIRT_ENV_NAME} not found..."
      exit 1
   fi
fi


# data-directories: make adaptions where required
# Base-directory of EURAD-IM data
data_basedir=<path_to_data>
# Output-directory where downscaling results will be saved
out_basedir=../results/
# Base-directory where trained models are saved
model_basedir=../trained_models/destine_final
# target variable of downscaling
target_var="NOx"                     # options: O3, NOx
# grid resolution of EURAD-IM input data (3 for DestinE use case simulations, 5 for archived EURAD-IM simulations)
grid_res=3                           # options: 3., 5.
# initialization time of EURAD-IM run
init_time="2018-07-24"               # any date-format that can be recognized pandas.to_datetime

# run job
srun --overlap python3 ${BASE_DIR}/main_scripts/main_inference.py -data_base_dir ${data_basedir} -output_base_dir ${out_basedir} -model_base_dir ${model_basedir} \
                                                                  -target_var ${target_var} -grid_res ${grid_res} -init_time ${init_time}

