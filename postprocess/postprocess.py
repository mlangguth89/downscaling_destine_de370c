# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Auxiliary methods for postprocessing.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-12-08"
__update__ = "2024-03-04"

import os
from typing import Union, List
import logging
import numpy as np
import xarray as xr
from statistical_evaluation import feature_importance, get_spectrum
from plotting import create_line_plot, create_map_score, create_box_plot, create_ps_plot

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
list_or_str = Union[List[str], str]

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)


def get_model_info(model_base, output_base: str, exp_name: str, bool_last: bool = False, model_type: str = None):

    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_model_info.__name__}")

    model_name = os.path.basename(model_base)
    norm_dir = model_base


    add_str = "_last" if bool_last else "_best"

    if "wgan" in exp_name:
        func_logger.debug(f"WGAN-modeltype detected.")
        model_dir, plt_dir = os.path.join(model_base, f"{exp_name}{add_str}", f"{exp_name}_generator{add_str}"), \
                             os.path.join(output_base, model_name)
        model_type = "sha_wgan"
    elif "unet" in exp_name or "deepru" in exp_name:
        func_logger.debug(f"U-Net-modeltype detected.")
        model_dir, plt_dir = os.path.join(model_base, f"{exp_name}{add_str}"), os.path.join(output_base, model_name)
        model_type = "sha_unet" if "unet" in exp_name else "deepru"
    else:
        func_logger.debug(f"Model type could not be inferred from experiment name. Try my best by defaulting...")
        if not model_type:
            raise ValueError(f"Could not infer model type from experiment name" +
                             "nor found parsed model type --model_type/model.")
        model_dir, plt_dir = model_base, os.path.join(output_base, model_name)
        norm_dir = model_dir
        model_type = model_type

    return model_dir, plt_dir, norm_dir, model_type


def run_feature_importance(ds: xr.DataArray, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                           ref_score: float, data_loader_opt: dict, plt_dir: str, patch_size = (6, 6)):
    """
    Run feature importance analysis and create box-plot of results
    :param ds: Unnormalized xr.Dataset with predictors and target variable
    :param predictors: List of predictor names
    :param varname_tar: Name of target variable
    :param model: Model object
    :param norm: Normalization object
    :param score_name: Name of score to compute feature importance
    :param ref_score: Reference score to normalize feature importance scores
    :param data_loader_opt: Data loader options
    :param plt_dir: Directory to save plot files
    :param patch_size: Patch size for feature importance analysis
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_feature_importance.__name__}")
    
    # get feature importance scores
    feature_scores = feature_importance(ds, predictors, varname_tar, model, norm, score_name, data_loader_opt, 
                                        patch_size=patch_size)
    
    rel_changes = feature_scores / ref_score
    max_rel_change = int(np.ceil(np.amax(rel_changes) + 1.))

    # plot feature importance scores in a box-plot with whiskers where each variable is a box
    plt_fname = os.path.join(plt_dir, f"feature_importance_{score_name}.png")

    create_box_plot(rel_changes.T, plt_fname, **{"title": f"Feature Importance ({score_name.upper()})", "ref_line": 1., "widths": .3, 
                                                 "xlabel": "Predictors", "ylabel": f"Rel. change {score_name.upper()}", "labels": predictors, 
                                                 "yticks": range(1, max_rel_change), "colors": "b"})

    return feature_scores


def run_evaluation_time(score_engine, score_name: str, score_unit: str, plot_dir: str, **plt_kwargs):
    """
    Create line plots of desired evaluation metric. Evaluation metric must have a time-dimension
    :param score_engine: Score engine object to comput evaluation metric
    :param score_name: Name of evaluation metric (must be implemented into score_engine)
    :param score_unit: Unit of evaluation metric
    :param plot_dir: Directory to save plot files
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_evaluation_time.__name__}")

    # create output-directories if necessary 
    metric_dir = os.path.join(plot_dir, "metric_files")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    
    model_type = plt_kwargs.get("model_type", "wgan")

    func_logger.info(f"Start evaluation in terms of {score_name}")
    score_all = score_engine(score_name)
    #score_all = score_all.drop_vars("variables")

    func_logger.info(f"Globally averaged {score_name}: {score_all.mean().values:.4f} {score_unit}, " +
                     f"standard deviation: {score_all.std().values:.4f}")  
    
    score_hourly_all = score_all.groupby("time.hour")
    score_hourly_mean, score_hourly_std = score_hourly_all.mean(), score_hourly_all.std()

    # create plots
    create_line_plot(score_hourly_mean, score_hourly_std, model_type.upper(),
                     {score_name.upper(): score_unit},
                     os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}.png"), **plt_kwargs)

    scores_to_csv(score_hourly_mean, score_hourly_std, score_name, fname=os.path.join(metric_dir, f"eval_{score_name}_year.csv"))

    score_seas = score_all.groupby("time.season")
    for sea, score_sea in score_seas:
        score_sea_hh = score_sea.groupby("time.hour")
        score_sea_hh_mean, score_sea_hh_std = score_sea_hh.mean(), score_sea_hh.std()
        func_logger.debug(f"Evaluation for season '{sea}'...")
        create_line_plot(score_sea_hh_mean,
                         score_sea_hh.std(),
                         model_type.upper(), {score_name.upper(): score_unit},
                         os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{sea}.png"),
                         **plt_kwargs)
        
        scores_to_csv(score_sea_hh_mean, score_sea_hh_std, score_name, 
                      fname=os.path.join(metric_dir, f"eval_{score_name}_{sea}.csv"))
    return score_all


def run_evaluation_spatial(score_engine, score_name: str, plot_dir: str, 
                           dims = ["rlat", "rlon"], **plt_kwargs):
    """
    Create map plots of desired evaluation metric. Evaluation metric must be given in rotated coordinates.
    :param score_engine: Score engine object to comput evaluation metric
    :param plot_dir: Directory to save plot files
    :param dims: Spatial dimension names
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{run_evaluation_time.__name__}")
    
    os.makedirs(plot_dir, exist_ok=True)

    model_type = plt_kwargs.get("model_type", "wgan")
    score_all = score_engine(score_name)
    #score_all = score_all.drop_vars("variables")

    score_mean = score_all.mean(dim="time")
    fname = os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_avg_map.png")
    create_map_score(score_mean, fname, dims=dims,
                     title=f"{score_name.upper()} (avg.)", **plt_kwargs)

    score_hourly_mean = score_all.groupby("time.hour").mean(dim=["time"])
    for hh in range(24):
        func_logger.debug(f"Evaluation for {hh:02d} UTC")
        fname = os.path.join(plot_dir, f"downscaling_{model_type}_{score_name.lower()}_{hh:02d}_map.png")
        create_map_score(score_hourly_mean.sel({"hour": hh}), fname,
                         dims=dims, title=f"{score_name.upper()} {hh:02d} UTC",
                         **plt_kwargs)

    for hh in range(24):
        score_now = score_all.isel({"time": score_all.time.dt.hour == hh}).groupby("time.season").mean(dim="time")
        for sea in score_now["season"]:
            func_logger.debug(f"Evaluation for season '{str(sea.values)}' at {hh:02d} UTC")
            fname = os.path.join(plot_dir,
                                 f"downscaling_{model_type}_{score_name.lower()}_{sea.values}_{hh:02d}_map.png")
            create_map_score(score_now.sel({"season": sea}), fname, dims=dims,
                             title=f"{score_name} {sea.values} {hh:02d} UTC", **plt_kwargs)

    return True


def run_spectral_analysis(ds: xr.Dataset, plt_dir: str, varname: str, var_unit: str,
                          lonlat_dims: list_or_str = ["rlon", "rlat"], lcutoff: bool= True, re: float = 6371.):
    """
    Run spectral analysis for all experiments provided in xr.Dataset and plot the data.
    """
    func_logger = logging.getLogger(f"{logger_module_name}.{run_spectral_analysis.__name__}")

    exps = list(ds.data_vars)
    nexps = len(exps)

    ps_dict = {}

    # compute wave numbers based on size of input data
    nlon, nlat = ds[lonlat_dims[0]].size, ds[lonlat_dims[1]].size

    dims = ["wavenumber"]
    coord_dict = {"wavenumber": np.arange(0, np.amin(np.array([int(nlon/2), int(nlat/2)])))}

    for i, exp in enumerate(exps):
        func_logger.info(f"Start spectral analysis for experiment {exp} ({i+1}/{nexps})...")

        # run spectral analysis
        ps_exp = get_spectrum(ds[exp], lonlat_dims = lonlat_dims, lcutoff= lcutoff, re=re)
        # average over all time steps and create xarray.DataArray
        da_ps_exp = xr.DataArray(ps_exp.mean(axis=0), dims=dims, coords=coord_dict, name=exp)

        # remove wavenumber 0 and append dictionary    
        ps_dict[exp] = da_ps_exp[1::]

    # create xarray.Dataset 
    ds_ps = xr.Dataset(ps_dict)

    # create plot   
    plt_fname = os.path.join(plt_dir, f"{varname}_power_spectrum.png")
    create_ps_plot(ds_ps, {varname: f"{var_unit}**2 m"}, plt_fname, colors= ["navy", "green"],
                   x_coord="wavenumber")

def scores_to_csv(score_mean, score_std, score_name, fname="scores.csv"):
    """
    Save scores to csv file
    :param score_mean: Hourly mean of score
    :param score_std: Hourly standard deviation of score
    :param score_name: Name of score
    :param fname: Filename of csv file
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{scores_to_csv.__name__}")
    
    df_mean = score_mean.to_dataframe(name=f"{score_name}_mean")
    df_std = score_std.to_dataframe(name=f"{score_name}_std")
    df = df_mean.join(df_std)

    func_logger.info(f"Save values of {score_name} to {fname}...")
    df.to_csv(fname)
