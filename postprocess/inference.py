# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Auxiliary methods for running inference on a trained model without further postprocessing.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2024-04-25"
__update__ = "2024-04-25"

# import packages
import os
from typing import List, Tuple, Dict
import logging
from hashlib import md5
from cdo import Cdo
import pandas as pd
from models.model_engine import ModelEngine

# auxiliary variable for logger
logger_module_name = f"main_inference.{__name__}"
module_logger = logging.getLogger(logger_module_name)

def get_trained_model(model_type: str, shape_in: Tuple, predictands: List[str], hparams_dict: Dict, model_dir: str):
    """
    Initialize model and load trained weights from disk. 
    This has two advantages over keras.load_model:
    - it allows for data-agnostic inference (different input shape compared to training)
    - initialization is quicker
    However, note that the first advantage only holds for models that are not dependent on the input shape such as convolutional models.
    :param model_type: Type of model to be loaded (see ModelEngine).
    :param shape_in: Shape of the input data.
    :param ds_dict: Dictionary with dataset configuration.
    :param hparams_dict: Dictionary with model configuration.
    :param model_dir: Directory where model weights are stored. File must be named "model_weights.h5".
    :return: Trained model.
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_trained_model.__name__}")
    
    # initialize model
    func_logger.debug(f"Instantiate {model_type}-model.")
    model_instance = ModelEngine(model_type)
    model = model_instance(shape_in[1:], predictands, hparams_dict, model_dir, "dummy")

    if model_type == "sha_wgan":
        trained_model = model.generator
    else:
        trained_model = model

    # build model and load weights
    trained_model.build(input_shape=shape_in)
    func_logger.debug(f"Load weights from '{os.path.join(model_dir, 'model_weights.h5')}'.")
    trained_model.load_weights(os.path.join(model_dir, "model_weights.h5"))
    
    return trained_model


def get_grid_descriptions(gdes_dir, grid_res):
    """
    Get grid description files for input, intermediate and target grid.
    Note that for 5km-simulations, the grid description for the last operational EURAD-IM 5km-simulation set-up is used. 
    By contrast, the grid description for the 3km-simulations corresponds to the current set-up in the DestinE-AQ use case.
    :param gdes_dir: Directory where grid description files are stored.
    :param grid_res: Grid resolution of the EURAD-IM data.
    :return: Tuple of grid description files for input, intermediate and target grid.
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_grid_descriptions.__name__}")
    
    gdes_in = os.path.join(gdes_dir, f"euradim_{grid_res:d}km_date")
    gdes_inter = os.path.join(gdes_dir, f"euradim_{grid_res:d}km_inter_date")
    gdes_tar = os.path.join(gdes_dir, f"euradim_1km_tar_date")

    if grid_res == 3:
        date_repl = "destine_inference"
    elif grid_res == 5:
        date_repl = "20161116-20181231"

    gdes_in = gdes_in.replace("date", date_repl)
    gdes_inter = gdes_inter.replace("date", date_repl)
    gdes_tar = gdes_tar.replace("date", date_repl)

    # check if all grid description files exist
    gdes_all = [gdes_in, gdes_inter, gdes_tar]
    for gdes in gdes_all:
        if not os.path.isfile(gdes):
            func_logger.error(f"Grid description file {gdes} not found.")
            raise FileNotFoundError(f"Grid description file {gdes} not found.")
    
    return gdes_in, gdes_inter, gdes_tar


def grid_description_to_str(fname, header=4):
    """
    Read grid description file and return content as concatenated string.
    :param fname: File name of grid description file.
    :param header: Number of header lines to be skipped.
    :return: Concatenated string of grid description file.
    """
    gdes = pd.read_csv(fname, sep="\n", header=header)
    concat_str = ""
    for val in gdes.values:
        concat_str += str(val).replace(" ", "")
    
    return concat_str.replace("[\'", "").replace("\']", "")


def create_weight_file(remap_method, data_file: str, gdes_in, gdes_out, weight_dir = None, header: int = 4):
    """
    Create weight file for remapping data from input to target grid with CDO.
    :param remap_method: Method for remapping (remapcon or remapbil).
    :param data_file: Data file to be remapped.
    :param gdes_in: Grid description file of input grid.
    :param gdes_out: Grid description file of target grid.
    :param weight_dir: Directory where weight file is stored. 
                       If None, weight file is stored in the same directory as the target grid description file.
    :param header: Number of header lines to be skipped in grid description files.
    :return: File name of weight file.
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{create_weight_file.__name__}")

    cdo = Cdo()

    gdes_hash = get_gdes_hash(gdes_in, gdes_out, header=header)
    
    dest_dir = weight_dir if weight_dir else os.path.dirname(gdes_out) 
    
    if remap_method == "remapcon":
        ofile = os.path.join(dest_dir, f"remapcon_{gdes_hash}.nc")
        
        func_logger.info(f"Write weight for remapping to '{ofile}'.")
        func_logger.debug(f"CDO-command: cdo gencon,{gdes_out} -setgrid,{gdes_in} {data_file} {ofile}")
        cdo.gencon(gdes_out, input=f"-setgrid,{gdes_in} {data_file}", output=ofile)
    elif remap_method == "remapbil":
        ofile = os.path.join(dest_dir, f"remapbil_{gdes_hash}.nc")
        
        func_logger.info(f"Write weight for remapping to '{ofile}'.")
        func_logger.debug(f"CDO-command: cdo genbil,{gdes_out} -setgrid,{gdes_in} {data_file} {ofile}")
        cdo.genbil(gdes_out, input=f"-setgrid,{gdes_in} {data_file}", output=ofile)    
    else: 
        raise ValueError(f"{remap_method} is unknown. Valid choices are: 'remapcon' and 'remapbil'")
        
    return ofile
        
def get_gdes_hash(*args, header=4):
    """
    Create md5-hash from grid description files.
    :param args: List of grid description files.
    :param header: Number of header lines to be skipped in grid description files.
    :return: Hash of grid description files.
    """
    gdes_str = ""
    for gdes in args:
        gdes_str += grid_description_to_str(gdes, header=header)
        
    hash_ = md5(gdes_str.encode('utf-8')).hexdigest()
        
    return hash_

        
def check_weight_file(remap_method, data_file: str, gdes_in, gdes_out, weight_dir = None, header: int = 4):
    """
    Check if weight file for remapping data from input to target grid exists.
    If not, create weight file.
    :param remap_method: Method for remapping (remapcon or remapbil).
    :param data_file: Data file to be remapped.
    :param gdes_in: Grid description file of input grid.
    :param gdes_out: Grid description file of target grid.
    :param weight_dir: Directory where weight file is stored. 
                       If None, weight file is stored in the same directory as the target grid description file.
    :param header: Number of header lines to be skipped in grid description files.
    :return: File name of weight file.
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{check_weight_file.__name__}")
    
    gdes_hash = get_gdes_hash(gdes_in, gdes_out)
    
    wgt_dir = weight_dir if weight_dir else os.path.dirname(gdes_out) 
    wgt_file = os.path.join(wgt_dir, f"{remap_method}_{gdes_hash}.nc")
    
    if os.path.isfile(wgt_file):
        func_logger.info(f"Found weight-file '{wgt_file}' for {remap_method}.")
    else:
        func_logger.info(f"Create new weight-file '{wgt_file}' for {remap_method}.")
        wgt_file = create_weight_file(remap_method, data_file, gdes_in, gdes_out, weight_dir, header)
        
    return wgt_file     
