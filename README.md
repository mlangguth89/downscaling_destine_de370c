# ML downscaling of EURAD-IM forecasts in Destine DE370c

## Background
In scope of the Destination Earth air quality use case (DE_370c), two Wasserstein Generative Networks (WGANs) have been trained to downscale nitrogen oxide (NOx) and ozone (O3) EURAD-IM forecasts to kilometre-scale. Training of these models have been done with archived EURAD-IM forecasts that are produced operationally at [IEK-8](https://www.fz-juelich.de/de/iek/iek-8/forschung/modellierung/modelle/eurad-im) since 2012. <br>
Although the accuracy of the attained models is not satisfactory due to a lack of informative predictors from the EURAD-IM forecast data, both models are publsihed as part of this reporsitory. Furthermore, the code to preprocess the EURAD-IM data as well as to train the WGAN models is provided here. A script to run inference on the trained WGANs is also available, ready for operational deployment.<br> 
Large parts of the code base have been forked from Application 5 of the [MAELSTROM project](https://www.maelstrom-eurohpc.eu/) avaialable at: [https://gitlab.jsc.fz-juelich.de/esde/machine-learning/downscaling_maelstrom](https://gitlab.jsc.fz-juelich.de/esde/machine-learning/downscaling_maelstrom).

## Software requirements

All scripts are written in Python and have been tested with Python 3.9.6. Newer Python versions are possible, but may require adaptions/updates of the following Python package versions:

- Tensorflow 2.6.0 (GPU-support recommended)
- numpy/1.21.3
- pandas/1.3.4
- xarray/0.20.1
- dask/2021.9.1
- netCDF4/1.5.7
- cdo/1.6.0
- pyproj/3.3.0
- matplotlib/3.4.3 (for Jupyter Notebook only)
- cartopy/0.20.0 (for Jupyter Notebook only)
- skimage/0.18.3 (for Jupyter Notebook only)

It's recommended to provide these Python package in a virtual environment. <br><br>

Note that these Python packages have further dependencies. In particular, `netCDF4` requires the [HDF5 C](https://www.hdfgroup.org/solutions/hdf5/) and the [netCDF C](https://www.unidata.ucar.edu/software/netcdf/) library, see [here](https://unidata.github.io/netcdf4-python/) for more details. CDO furthermore requires PROJ (also required by the `pyproj` Python-package), ecCodes and further external libraries, see [here](https://code.mpimet.mpg.de/projects/cdo/wiki#Download-Compile-Install). Finally, [NCO](https://nco.sourceforge.net/) must also be available. <br>
A list of external library versions that are known to work includes:
- NetCDF/4.8.1
- hdf5/1.12.1
- CDO/2.0.2
- ecCodes/2.22.1
- proj/8.1.0
- NCO/5.0.3

On the **Juelich HPC-systems**, the virtual environment can be conveniently created with the script `env_setup/create_env.sh`. 
```
> cd env_setup
> source create_env.sh <virtual_env_name>
```
This script will load all modules listed in `env_setup/modules_jsc.sh`, minimizing the demand for custom installation of software.<br> On other HPC-systems (such as LUMI), the scripts may be adapted according to the available software module stack. 

## Run (operational) inference on trained models 

Two trained WGANs for downscaling NOx and ozone are provided in this reporsitory. The models are saved in the `trained_models/destine_final`. Albeit the accuracy of these downscaling models has not reached a satisfactory level, a script for operational deployment for demonstration purposes is made available. <br>
Inference on the trained models can be run by  teh script `main_scripts/main_inference.py`. This is called by:
```
> cd main_scripts
> python main_inference.py [-h] [--data_base_directory/-data_base_dir PATH_TO_DATA] [--output_base_directory/-output_base_dir PATH_TO_OUTPUT] [--model_base_directory/-model_base_dir MODEL_DIR] [--initialization_time/-init_time INIT_TIME] [--target_variable/-target_var TARGET_VAR] [--grid_resoultion/-grid_res GRID_RES] [--time_steps/-time_steps TIME_STEPS] [--with_gu/-gpu] 
``` 

The arguments denote the following:

| short | long | default | help | 
| ------ | ------ | ------ | ------ |
| -h     | --help |        | shows this help message and exit      |
| -data_base_dir  | --data_base_directory      | `None` | Top-level under which EURAD-IM simulations are stored (in sub-directories). |
| -output_base_dir| --output_base_directory    | `None` | Directory where netCDF-file with downscaling results will be saved. |
| -model_base_dir | --model_base_directory     | `../trained_models/destine_final` | Base directory where trained models are saved. |
| -init_time      | --initialization_time      | `None` | Initialization time of EURAD-IM simualtion that will be downscaled. |
| -grid_res       | --grid_resoultion          | `3`    | Grid resolution/spacing of the EURAD-IM data to downscale (options: [3, 5]) |
| -target_var     | --target_variable          | `NOx`  | Name of the target variable to downscale (options: ["NOx", "O3"]) |
| -time_steps     | --time_steps               | `2/25` | Time steps of the EURAD-IM data to be used for downscaling. |
| -gpu            | --with_gpu                 |        | Flag to run inference on GPU. |

A particular example to downscale an EURAD IM forecast run from the use case may look as follows:

```
> cd main_scripts
> python main_inference.py -data_base_dir <path_to_euradim_data> -output_base_dir ../results \ 
                           -init_time 2018-07-25 -target_var NOx -grid_res 3 -time_steps 2/49
```
The EURAD-IM forecast data to be downscaled must be provided in a netCDF-file that is located in a subdirectory `<path_to_euradim_data>/YYYYMM/DD/` where YYYYMM and DD represent the year-month and the day of the forecast run's initialization time (cf. `-init_time`-argument). In this particular example, a file `ctmout_*de3.nc` must be available under `<path_to_euradim_data>/201807/25`. <br> 
The suffix `de3` denotes the domain of the EURAD-IM simulation that is centered over Germany with a grid spacing of 3 km (cf. `-grid_res`-argument) as used in this use case. Additionally, EURAD-IM simulations with a grid spacing of 5 km that have been operational between 2016/11/16 and 2018/12/31 can be downscaled by parsing `-grid_res 5`. Note that the required CDO grid descriptions for data remapping are stored under the `grid_des/`-directory of this reporsitory. Other domain configurations are not supported. <br>
As indicated by the argument `-target_var`, the WGAN for downscaling NOx-data will be used (parse `-target_var O3` for downscaling ozone data). The downscaled data will be saved in a netCDF-file `downscaling_nox_2018072500.nc` under the `results/` directory as controlled by the `-output_base_dir`-argument. The forecast data for the time steps between 2 and 49, corresponding to lead times of one and 48 hours, respectively, will be downscaled (see `-time_steps`-argument). <br>
Optionally, the inference script can be run on a GPU (if available) by adding the flag `--with_gpu/-gpu`. <br><br>
For the **JSC HPC-system JURECA**, a batch script template to submit the inference job on the CPU-nodes is provided under `HPC_batch_scripts`. The corresponding batch-script can also be used as blueprint for job-submission on other HPC-systems (such as LUMI).

## Statistical evaluation of the trained models 

The Jupyter Notebook `postprocess_euradim.ipynb` can be used to evaluate the two WGANs for downscaling NOx and O3 data. 
```
> cd main_scripts
> jupyter-notebook postprocess_euradim.ipynb &
```
The Notebook is available unde the `main_scripts/`-directory. Note the extended software requirements for the Jupyter Notebook kernel listed [above](#software-requirements).

## Train downscaling models from scratch

The scripts `main_preprocessing.py` and `main_train.py` can be adapted to develop ML downscaling solutions with a more solid database, i.e. a consistent and comprehensive set of EURAD-IM hindcast simulations providing more predictor variables such as local emission information. The required adaptions are briefly outlined subsequently:

### Preprocessing the EURAD-IM data 

The script `main_preprocessing.py` allows a customized configuration of the data preprocessing from archived EURAD-IM forecasts. The forecast data must be provided in netCDF-format where one  file per model run providing hourly forecasts with a lead time up to 24 hours is expected. The netCDF-files must be available in monthly and daily subdirectories with the relative path YYYYMM/DD where YYYY, MM and DD represent the year, month and day of the simulation’s initial date.
The following arguments are used to configure the preprocessing:
| short | long | default | help | 
| ------ | ------ | ------ | ------ |
| -h     | --help |        | shows this help message and exit      |
| -predictors  | --predictors      | `None` | Dictionary where the keys denote the name of the predictor variable in the netCDf-file and the values provide the list of model levels from which data should be used. Example: `{“NO2”: [1]}` would extract NO2-data at model level 1. |
| -predictands| --predictands    | `None` | like --predcitors/-predictors, but for the predictand variables |
| -tar_datadir | --target_datadir     | `None` | Top-level directory where the EURAD-IM simulations are saved. The netCDf-files must be located in monthly and daily subdirectories. Note also that coarse-grained input and high-resolved target datafiles must be available in the same directory. In case that both files are located in different directories, `--input_datadir`/ `-in_datadir` must be used additionally. |
| -out_dir      | --preproc_outdir      | `None` | Top-level directory under which preprocessed data (monthly netCDF-files) will be stored. |
| -grid_des_tar       | --grid_description_target          | `../grid_des/`    | Directory where CDO grid descriptions for input and target data are available. Furthermore, a grid description for the conservative remapping step is required (coarse grid aligned with high-resolution grid) |
| -y     | --years          | `None`  | List of years for which data shall be preprocessed. |
| -m     | --months               | `all` | List of months for which data shall be preprocessed. “all” ensures that the data for all months is preprocessed. |
| -down_fac           | --downscaling_factor                 |   `5`    | Downscaling factor between input and target data. Possible choices for EURAD-IM data are 3 and 5 corresponding to EURAD-IM simulations on a 5 km- and 3 km-grid, respectively. |

**Caveats**:  
- Configuring the model-level for the predictor and predictand variables is not supported yet. The selected model-level is hard-coded to level 1 (see l. 196 and l.226 in preprocess_euradim_forecasts.py 
- Log-transformation of the data as well as the residual approach require manual processing on the netCDF-files resulting from the preprocessing. Both can be realized with the help of CDO’s aexpr-operator, while the name of the modified variables should be appended with a “ln”-prefix or a “_res”-suffix. The corresponding CDO-commands may look as follows:
- ```
  cdo aexpr,”lnNOx_in=log(NOx_in+0.01)-log(0.01)” <infile> <outfile>
  ``` 
  applies the log-transformation on the input NOx-data.
- ```
  cdo aexpr,”O3_res_tar=O3_tar-O3_in” <infile> <outfile>
  ```
  calculates the resiudals of the target O3 data.


### Training a new downscaling model 
The following arguments are used to configure the script `main_train.py` that will train a model from scratch:

| short | long | default | help | 
| ------ | ------ | ------ | ------ |
| -conf_md | --configuration_model | `None` | JSON-file to configure the model architecture and the training (e.g. number of epochs, pretraining epochs and learning rate schedule). |
| -conf_ds | --configuration_dataset| `None` | JSON-file to configure the dataset for training. Here, the predictors and predictands are defined as exemplified in the `config_ds_euradim_o3.json` and `config_ds_euradim_nox.json` (see `config`-directory of the repository). |
| -in | --input_dir| `None` | Directory where the preprocessed EURAD-IM data is located. The naming pattern for the training data files must be `downscaling_euradim_*_train.nc` and `downscaling_euradim_*_val.nc`  for the validation data files. |
| -out | --output_dir | `None` | Output directory where model is saved. |
| -model | --downscaling_model | `None` | Name of the model architecture for downscaling (`"sha_wgan"` has been used in this use case) |
| -exp_name |--experiment_name | `None` | Custom name for the current experiment | 
| -js_norm | --json_norm_file | `None` | JSON-file providing normalization parameters. If not provided, the parameters are derived from the training dataset. |
| -id | --job_id | `None` | Job-id from Slurm. |

