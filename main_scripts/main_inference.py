# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference with trained downscaling models on EURAD-IM output.
Only the downscaled data is saved to a netCDF-file without further postprocessing.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-04-22"
__update__ = "2024-04-25"

# import packages
import os, glob
import logging
import argparse
from timeit import default_timer as timer
from typing import Tuple
import json as js
from cdo import Cdo 
import numpy as np
import pandas as pd
import tensorflow as tf
from all_normalizations import ZScore
from other_utils import finditem, convert_to_xarray, config_logger
from preprocess_euradim_forecasts import PreprocessEURADIM
from inference import get_grid_descriptions, check_weight_file, get_trained_model

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.DEBUG)


def main(parser_args):

    t0 = timer()

    log_file = os.path.join("./", f"inference_sha_wgan_{parser_args.target_var.lower()}.log")
    logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
    logger = config_logger(logger, log_file)  

    # parameter
    eps = 0.01              # epsilon for log-transformation

    # Initialize CDO
    cdo = Cdo()

    # Choose device
    if parser_args.with_gpu:
        device = "/gpu:0"
    else:
        device = "/cpu:0"
    
    logger.info(f"Inference will be run on device {device}.")

    # Set up auxiliary variables and paths to directories
    model_type = "sha_wgan"                                                     # only WGAN with tuned Sha U-Net is provided
    model_name = f"{model_type}_{parser_args.target_var.lower()}"
    model_base_dir = os.path.join(parser_args.model_base_dir, model_name)
    model_dir = os.path.join(model_base_dir, f"{model_name}_generator")

    init_time = parser_args.init_time
    euradim_data_dir = os.path.join(parser_args.data_base_dir, init_time.strftime("%Y%m"), init_time.strftime("%d"))

    # Check availability of EURAD-IM data
    if parser_args.grid_res == 3:
        pattern = "ctmout_digitwin_*de3.nc"
    else:
        pattern = "ctmout_fc05c_*_j05.nc"
        
    euradim_file = glob.glob(os.path.join(euradim_data_dir, pattern))

    if not euradim_file:
        logger.error(f"EURAD-IM data for {init_time} not found in {euradim_data_dir}.")
        raise FileNotFoundError(f"EURAD-IM data for {init_time} not found in {euradim_data_dir}.")
    else:
        euradim_file = euradim_file[0]

    # Get dataset and model/hyperparameter configuration as well as the normalization file
    try:
        t0_prepare = timer()
        logger.info(f"Loading configuration files from {model_base_dir}.")

        ds_config = os.path.join(model_base_dir, f"config_ds_euradim_{parser_args.target_var.lower()}.json")
        logger.info(f"Loading dataset configuration from {ds_config}.")
        with open(ds_config, "r") as dsf:
            ds_dict = js.load(dsf)

        md_config = os.path.join(model_base_dir, f"config_{model_type}.json")
        logger.info(f"Loading model configuration from {md_config}.")
        with open(md_config, "r") as mdf:
            hparams_dict = js.load(mdf)

        js_norm = os.path.join(model_base_dir, "norm.json")
        logger.info(f"Loading normalization configuration from {js_norm}.")
        norm_obj = ZScore(ds_dict["norm_dims"])
        norm_obj.read_norm_from_file(js_norm)

        logger.info(f"Model and configuration files loaded successfully in {timer() - t0_prepare:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error loading configurations from {model_base_dir}.")
        raise e
    
    logger.info(f"Initialization took {timer() - t0:.2f} seconds.")

    ### Preprocess EURAD-IM data ###
    t0_remap = timer()
    logger.info(f"Preprocessing EURAD-IM data from {euradim_file}.")

    # get grid description files
    gdes_in, gdes_inter, gdes_tar = get_grid_descriptions(os.path.join(parser_args.model_base_dir, "..", ".." , "grid_des"), parser_args.grid_res)

    # get predictor variable and handle log-transformed variables
    predictors = list(ds_dict["predictors"])                                        # all input predictors for downscaling model
    log_predictors = [var for var in predictors if var.startswith("ln")]            # log-trasnformed input predictors
    predictors_raw = [var.lstrip("ln").rstrip("_in") for var in predictors]         # predictors as available from EURAD-IM data

    predictor_str = ",".join(predictors_raw)

    tar_varname = ds_dict["predictands"][0]   
    
    # construct argument-string for CDO
    cdo_args = f"-selname,{predictor_str} -sellevidx,1 -seltimestep,{parser_args.time_steps} {euradim_file}"

    if "NOx" in predictors_raw:
        cdo_args = cdo_args.replace("-sellevidx,1", f"{PreprocessEURADIM.get_cdo_string_nox(predictor_str)} -sellevidx,1")

    # remap input data to intermediate grid
    wgt_remapcon_file = check_weight_file("remapcon", euradim_file, gdes_in, gdes_inter)
    logger.info(f"Remapping input data (file: '{gdes_in}') to intermediate grid (file: '{gdes_inter}').")
    ofile = cdo.remap(f"{gdes_inter},{wgt_remapcon_file}", input=f"{cdo_args}", options="-L")

    # remap to target grid 
    wgt_remapbil_file = check_weight_file("remapbil", ofile, gdes_inter, gdes_tar)
    logger.info(f"Bi-linearly interpolate intermediate data (file: '{gdes_inter}') to target grid (file: '{gdes_tar}').")
    ds = cdo.remap(f"{gdes_tar},{wgt_remapbil_file}", input=f"-setgrid,{gdes_inter} {ofile}", returnXDataset = True, options="-L")

    logger.info(f"Preprocessing completed in {timer() - t0_remap:.2f} seconds.")

    ds = ds.squeeze()      # remove singleton dimensions

    # Steps to normalize and transform data
    t0_norm = timer()
    
    # rename variables for normalization
    rename_dict = {}

    for key, value in zip(predictors_raw, predictors):
        # Assign key-value pairs to the dictionary
        rename_dict[key] = value

    ds = ds.rename(rename_dict)

    # apply log-transformation to predictors if required
    for var in log_predictors:
        ds[var] = np.log(ds[var] + eps) - np.log(eps)
        
    # set flags for handling residual and inverse log-transformation
    if "_res" in tar_varname:
        lresidual_app = True
        varname_base = tar_varname.split("_")[0]
        varname_in = f"{varname_base}_in"
        
        y_corr = ds[varname_in]
    else: 
        lresidual_app = False
        
    llog = False
    if f"ln" in tar_varname: llog = True

    logger.debug(f"Residual approach on predictand {tar_varname}? {lresidual_app}")
    logger.debug(f"Log transformed predictand {tar_varname}? {llog}")

    # normalize data
    ds = norm_obj.normalize(ds)
    logger.info(f"Data normalization completed in {timer() - t0_norm:.2f} seconds.")

    ### Initialize and load model ###
    t0_model = timer()

    coords = ds.coords
    ds = ds[predictors].to_array().transpose(..., "variable")
    input_shape = ds.shape

    try:
        with tf.device(device):
            trained_model = get_trained_model(model_type, input_shape, ds_dict["predictands"], hparams_dict, model_dir)
    except Exception as e:
        logger.error(f"Error loading model from {model_dir}.")
        raise e

    logger.info(f"Model loaded successfully in {timer() - t0_model:.2f} seconds.")

    ### Perform inference ###
    logger.info(f"Performing inference on EURAD-IM data.")
    t0_inference = timer()

    with tf.device(device):
        y_pred = trained_model.predict(ds.values, verbose=2)

    logger.info(f"Inference completed in {timer() - t0_inference:.2f} seconds.")

    ### Denormalize data ###
    t0_postproc = timer()

    print(f"Denormalizing downscaling results.")
    tar_varname = ds_dict["predictands"][0] 
    y_pred = convert_to_xarray(y_pred, norm_obj, tar_varname, coords,
                               ["time", "y", "x"], finditem(trained_model.hparams, "z_branch", False))
    
    # apply inverse log-transformation to predictand if required
    if lresidual_app: y_pred = y_pred + y_corr
    if llog: y_pred = eps*np.exp(y_pred)-eps

    # Save downscaling results to netCDF-file
    ncfile_out = os.path.join(parser_args.output_base_dir, f"downscaling_{parser_args.target_var.lower()}_{init_time.strftime('%Y%m%d%H')}.nc")
    os.makedirs(parser_args.output_base_dir, exist_ok=True)
    logger.info(f"Write downscaling results to netCDF-file.")

    ds = y_pred.to_dataset(name=parser_args.target_var)
    ds.to_netcdf(ncfile_out)

    logger.info(f"Denormalization and saving of downscaling results completed in {timer() - t0_postproc:.2f} seconds.")
    logger.info(f"Total runtime: {timer() - t0:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_directory", "-data_base_dir", dest="data_base_dir", type=str, required=True,
                        help="Top-level under which EURAD-IM simulations are stored (in sub-directories).")
    parser.add_argument("--output_base_directory", "-output_base_dir", dest="output_base_dir", type=str, required=True,
                        help="Directory where netCDF-file with downscaling results will be saved.")
    parser.add_argument("--model_base_directory", "-model_base_dir", dest="model_base_dir", type=str, default="../trained_models/destine_final/",
                        help="Base directory where trained models are saved.")
    parser.add_argument("--initialization_time", "-init_time", dest="init_time", type=lambda d: pd.to_datetime(d), required=True,
                        help="Initialization time of EURAD-IM simualtion that will be downscaled.")
    parser.add_argument("--target_variable", "-target_var", dest="target_var", type=str, default="NOx",
                        choices=["NOx", "O3"], help="Name of the target variable to downscale.")
    parser.add_argument("--grid_resolution", "-grid_res", dest="grid_res", type=int, default=3, choices=[3, 5],
                        help="Grid resolution/spacing of the EURAD-IM data.")
    parser.add_argument("--time_steps", "-time_steps", dest="time_steps", type=str, default="2/25",
                        help="Time steps of the EURAD-IM data to be used for downscaling. Default: 2/25.")
    parser.add_argument("--with_gpu", "-gpu", dest="with_gpu", default=False, action="store_true",
                        help="Flag to run inference on GPU.")

    args = parser.parse_args()
    main(args)
