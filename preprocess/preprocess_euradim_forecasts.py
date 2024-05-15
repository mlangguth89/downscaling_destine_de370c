# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2024-02-22"
__update__ = "2024-02-28"

"""
Class to preprocess EURA-Im forecasts available on JUCEDA. Preprocessing is supported for 5km- and 3km-simulations as input and 1 km-data as target.
The 5 km-data is available for the years 2012-2018, whereas the 3 km-simulations are operational since January 2019.

NOTE:
    * No distributed computing is implemented yet.
    * Data selection on specific model levels not supported yet, i.e. predictors = {"VAR": None} is equivalent to predictors = {"VAR": 10}
"""

import os, glob
import sys
import logging
from typing import Union, List
import traceback as tb
import datetime as dt
from cdo import *                                                                 # use CDO's Python bindings
import pandas as pd
from abstract_preprocess import AbstractPreprocessing, CDOGridDes
from other_utils import last_day_of_month, remove_files, to_list

# get logger
logger = logging.getLogger(os.path.basename(__file__).rstrip(".py"))
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')



class PreprocessEURADIM(AbstractPreprocessing):
    
    def __init__(self, tar_datadir: str, predictors: dict, predictands: dict, out_dir: str, 
                 grid_des_tar: str, downscaling_fac: int = 5, tmp_dir: str = "/work/langguth/.cdo_tmp",
                 log_file: str = None):
        """
        :param tar_datadir: path to directory of EURAD-IM forecasts (must contain input and target data!)
        :param predictors: dictionary with input data to be preprocessed
        :param predictands: dictionary with target data to be preprocessed
        :param out_dir: path to directory for preprocessed data
        :parama grid_des_tar: directory (!) where grid description files for EURAD-IM are located
        :param downscaling_fac: downscaling factor (5 or 3)
        :param tmp_dir: path to directory for temporary files created by CDO 
        :param log_dir: absolute path to log file
        """
        super().__init__("preprocess_euradim", tar_datadir, tar_datadir, predictors, predictands, out_dir)

        self.gdes_dir = grid_des_tar
        # check if downscaling factor is valid
        if downscaling_fac not in [3, 5]:
            raise ValueError("Downscaling factor must be 3 or 5!")
        self.downscaling_fac = downscaling_fac
        self.downscaling_task = "real"
        # grid descriptions
        self.grid_periods_all = ["20120101-20161115", "20161116-20181231", "20190101-20231231"]
        self.gdes_in, self.gdes_inter, self.gdes_tar = self.get_gdes_files(self.gdes_dir)

        self.predictor_str = ",".join(predictors.keys())              # string of predictors to be be read from input datafiles with CDO
        self.predictand_str = ",".join(predictands.keys())                    
        self.cdo = Cdo(tempdir=tmp_dir)                               # initialize CDO object

        # set-up logger (since it is not available without parallelization with PyStager)
        self.setup_logger(log_file)

    def __call__(self, *args, **kwargs):
        """
        Overwrite __call__ method of AbstractPreprocessing class
        """
        years, months = args
        years = years if isinstance(years, list) else [years]
        months = list(range(1, 13)) if months == ["all"] else months if isinstance(months, list) else to_list(months)

        print(months)
        for year in years:
            for month in months:
                logger.info(f"Preprocessing data for {year}-{month:02d}...")
                try:
                    self.preprocess(pd.Timestamp(f"{year}-{month:02}"))
                except Exception as err:
                    logger.exception(f"Preprocessing of data for {year}-{month:02d} failed! Error messgae:")


    # dummy methods for prepare_worker and preprocess_worker (since distributed computing is not yet implemented for JUCEDA)
    def prepare_worker(self):
        pass

    def preprocess_worker(self):
        pass

    # methods for preprocessing
    def get_gdes_files(self, gdes_dir: str):
        """
        Get and check grid description files for input, intermediate and target data are available
        :param gdes_dir: path to directory containing grid description files
        :return: paths to grid description files for input, intermediate and target data as dictionaries (since several grids have been operational)
        """

        if self.downscaling_fac == 5:
            grid_periods = self.grid_periods_all[0:2]
        elif self.downscaling_fac == 3:
            grid_periods = [self.grid_periods_all[2]]
    
        # initialize dictionaries for grid description files
        gdes_in, gdes_inter, gdes_tar  = {}, {}, {}

        for grid_period in grid_periods:
            # find grid description files for input, intermediate and target data and raise error if not found
            gdes_in[grid_period] = glob.glob(os.path.join(gdes_dir, f"euradim_{self.downscaling_fac:d}km_{grid_period}"))[0]
            gdes_inter[grid_period] = glob.glob(os.path.join(gdes_dir, f"euradim_{self.downscaling_fac:d}km_inter_{grid_period}"))[0]
            gdes_tar[grid_period] = glob.glob(os.path.join(gdes_dir, f"euradim_1km_tar_{grid_period}"))[0]

            if not gdes_in[grid_period]:
                miss_file = os.path.join(gdes_dir, f"feuradim_{self.downscaling_fac:d}km_{grid_period}")
                raise FileNotFoundError(f"No grid description file for input data for period {grid_period} ({miss_file}) found in {gdes_dir}!")
            if not gdes_inter[grid_period]:
                miss_file = os.path.join(gdes_dir, f"feuradim_{self.downscaling_fac:d}km_inter_{grid_period}")
                raise FileNotFoundError(f"No grid description file for intermediate for period {grid_period} ({miss_file}) data found in {gdes_dir}!")
            if not gdes_tar[grid_period]:
                miss_file = os.path.join(gdes_dir, f"feuradim_1km_tar_{grid_period}")
                raise FileNotFoundError(f"No grid description file for target data for period {grid_period} ({miss_file}) found in {gdes_dir}!")
            
        return gdes_in, gdes_inter, gdes_tar

    def preprocess(self, month: dt.datetime):
        
        day_range = pd.date_range(month.strftime("%Y-%m-01"), last_day_of_month(month), freq="D")
        nfails = 0
        fmt_month = "%Y-%m"

        for day in day_range:

            if nfails > 3:
                raise ValueError(f"Preprocessing of input and target data for {month.strftime(fmt_month)} failed for more than 3 days!")
            
            logger.info(f"Preprocessing data for {day.strftime('%Y-%m-%d')}...")
                  
            # process input data
            try:
               self.process_input_file(day)
            except Exception:
                logger.exception(f"Preprocessing of input data for {day.strftime('%Y-%m-%d')} failed! Error message:")
                self.ensure_clean_temp_files(day)
                nfails += 1
                continue
            # process target data
            try:
                self.process_target_file(day)
            except Exception:
                logger.exception(f"Preprocessing of target data for {day.strftime('%Y-%m-%d')} failed! Error message:")
                self.ensure_clean_temp_files(day)
                nfails += 1

        # merge preprocessed input and target files 
        logger.info(f"Merge preprocessed input and target files for {month.strftime(fmt_month)}...")       
        processed_infile = self.merge_input_files(month)
        processed_tarfile = self.merge_target_files(month)

        # merge input and target files  
        logger.info(f"Merge preprocessed input and target files for {month.strftime(fmt_month)}...")
        tar_var = list(self.predictands.keys())[0]
        logger.debug(f"Executed CDO-command: cdo merge {processed_infile} {processed_tarfile} {os.path.join(self.target_dir, f'downscaling_euradim_{tar_var}_{month.strftime(fmt_month)}.nc')}")
        self.cdo.merge(input=f"{processed_infile} {processed_tarfile}",
                       output=os.path.join(self.target_dir, f"downscaling_euradim_{tar_var}_{month.strftime(fmt_month)}.nc"))
        
        # remove temp files
        #self.clean_temp_files(month)

    def process_input_file(self, day: str):
        """
        Preprocess input data
        :param day: date for which to preprocess input data
        """
        datadir_now = os.path.join(self.source_dir_in, day.strftime("%Y%m"), day.strftime("%d"))
        datafile_in = glob.glob(os.path.join(datadir_now, f"ctmout*[p,j,h]{self.downscaling_fac:02d}*.nc"))[0]

        # output file and directory
        grid_period = self.get_grid_period(day)

        out_dir = os.path.join(self.target_dir, f"tmp_{day.strftime('%Y%m')}", f"input_{grid_period}")
        os.makedirs(out_dir, exist_ok=True)
        datafile_out = os.path.join(out_dir, f"input_{day.strftime('%Y%m%d')}.nc")
        
        # check if input data file exists
        if not os.path.exists(datafile_in):
            raise FileNotFoundError(f"No input data file found in {datadir_now}!")
        
        # slice data to variables of interest 
         # To-Do: don't hard-code level index and time selection
        logger.info(f"Processing input datafile {datafile_in}...")
        cdo_input_arg = f"-sellevidx,1 -seltimestep,2/25 {datafile_in}"

        if "NOx" in self.predictor_str:
            cdo_input_arg = f"{self.get_cdo_string_nox(self.predictor_str)} {cdo_input_arg}"
        
        logger.debug(f"Executed CDO-command: cdo selname {self.predictor_str} {cdo_input_arg} {datafile_out}")
        self.cdo.selname(self.predictor_str, input=cdo_input_arg, output=datafile_out)

    def process_target_file(self, day: str):
        """
        Preprocess target data
        :param day: date for which to preprocess target data    
        """
        datadir_now = os.path.join(self.source_dir_in, day.strftime("%Y%m"), day.strftime("%d"))
        datafile_in = glob.glob(os.path.join(datadir_now, "ctmout*[p,j,h]01*.nc"))[0]

        grid_period = self.get_grid_period(day)

        # output file and directory
        out_dir = os.path.join(self.target_dir, f"tmp_{day.strftime('%Y%m')}", f"target_{grid_period}")
        os.makedirs(out_dir, exist_ok=True)
        datafile_out = os.path.join(out_dir, f"target_{day.strftime('%Y%m%d')}.nc")
        
        # check if target data file exists
        if not os.path.exists(datafile_in):
            raise FileNotFoundError(f"No target data file found in {datadir_now}!")
        
        # slice data to variables of interest 
        # To-Do: don't hard-code level index and time selection
        logger.info(f"Processing target datafile {datafile_in}...")
        cdo_input_arg = f"-sellevidx,1 -seltimestep,2/25 {datafile_in}"

        if "NOx" in self.predictand_str:
            cdo_input_arg = f"{self.get_cdo_string_nox(self.predictand_str)} {cdo_input_arg}"
        
        logger.debug(f"Executed CDO-command: cdo selname {self.predictor_str} {cdo_input_arg} {datafile_out}")
        self.cdo.selname(self.predictand_str, input=cdo_input_arg, output=datafile_out)

    def merge_input_files(self, month: Union[str, dt.datetime]):
        """
        Merge preprocessed input files
        :param month: month for which to merge input files
        """
        input_subdirs = [f.path for f in os.scandir(os.path.join(self.target_dir, f"tmp_{month.strftime('%Y%m')}")) if f.is_dir() and f.name.startswith("input")]
        nsubdirs = len(input_subdirs)

        merged_infile = os.path.join(self.target_dir, f"tmp_{month.strftime('%Y%m')}", f"input_{month.strftime('%Y%m')}.nc")

        ofiles = []

        for subdir in input_subdirs:
            logger.info(f"Merge and remap input files in {subdir}...")
            grid_period = os.path.basename(subdir.rstrip("/")).strip("input_")

            logger.debug(f"Executed CDO-command: cdo mergetime {subdir}/*.nc tmp.nc")
            ofile = self.cdo.mergetime(input=f"{subdir}/*.nc")
            # remap to intermediate grid
            logger.debug(f"Executed CDO-command: cdo remapcon {self.gdes_inter[grid_period]} -setgrid,{self.gdes_in[grid_period]} tmp.nc tmp2.nc")
            ofile = self.cdo.remapcon(self.gdes_inter[grid_period], input=f"-setgrid,{self.gdes_in[grid_period]} {ofile}")
            # remap to target grid
            logger.debug(f"Executed CDO-command: cdo remapbil {self.gdes_tar[grid_period]} tmp2.nc tmp3.nc")
            ofiles.append(self.cdo.remapbil(self.gdes_tar[grid_period], input=f"-setgrid,{self.gdes_inter[grid_period]} {ofile}"))

        if nsubdirs > 1:
            logger.info(f"Merge input files from {nsubdirs} subdirectories...")
            logger.debug(f"Executed CDO-command: cdo mergetime <list of input files> tmp.nc")
            ofile = self.cdo.mergetime(input=ofiles)
        else:
            ofile = ofiles[0]

        # rename variables
        rename_str = ",".join([f"{var},{var}_in" for var in self.predictors.keys()])
        logger.debug(f"Executed CDO-command: cdo chname {rename_str} tmp.nc {merged_infile}")
        self.cdo.chname(rename_str, input=ofile, output=merged_infile)

        # remove temp files
        logger.debug("Clean CDO's temp directory.")
        self.cdo.cleanTempDir()

        return merged_infile
    
    def merge_target_files(self, month: Union[str, dt.datetime]):
        """
        Merge preprocessed target files
        :param month: month for which to merge target files
        """
        target_subdirs = [f.path for f in os.scandir(os.path.join(self.target_dir, f"tmp_{month.strftime('%Y%m')}")) if f.is_dir() and f.name.startswith("target")]
        nsubdirs = len(target_subdirs)

        ofiles = []

        for subdir in target_subdirs:
            logger.info(f"Merge target files in {subdir}...")
            grid_period = os.path.basename(subdir.rstrip("/")).strip("target_")

            logger.debug(f"Executed CDO-command: cdo mergetime {subdir}/*.nc tmp.nc")
            ofile = self.cdo.mergetime(input=f"{subdir}/*.nc")
            # remap to target grid
            logger.debug(f"Executed CDO-command: cdo remapbil {self.gdes_tar[grid_period]} tmp.nc tmp2.nc")
            ofiles.append(self.cdo.remapbil(self.gdes_tar[grid_period], input=ofile))

        if nsubdirs > 1:
            logger.info(f"Merge target files from {nsubdirs} subdirectories...")
            logger.debug(f"Executed CDO-command: cdo mergetime <list of target files> tmp.nc")
            ofile = self.cdo.mergetime(input=ofiles)
        else:
            ofile = ofiles[0]

        merged_tarfile = os.path.join(self.target_dir, f"tmp_{month.strftime('%Y%m')}", f"target_{month.strftime('%Y%m')}.nc")

        # rename variables
        rename_str = ",".join([f"{var},{var}_tar" for var in self.predictands.keys()])
        logger.debug(f"Executed CDO-command: cdo chname {rename_str} tmp.nc {merged_tarfile}")
        self.cdo.chname(rename_str, input=ofile, output=merged_tarfile)

        # remove temp files
        logger.debug("Clean CDO's temp directory.")
        self.cdo.cleanTempDir()

        return merged_tarfile
    
    def clean_temp_files(self, month:dt.datetime):
        """
        Remove temporary files
        :param month: month for which to remove temporary files
        """
        # get list of temporary files...
        target_subdirs = [f.path for f in os.scandir(os.path.join(self.target_dir, f"tmp_{month.strftime('%Y%m')}")) if f.is_dir()]

        file_list = list(glob.glob(os.path.join(self.target_dir, f"tmp_{month.strftime('%Y%m')}")))
        for subdir in target_subdirs:
            file_list.extend(glob.glob(os.path.join(subdir)))

        # ... and remove them
        remove_files(file_list)

    def ensure_clean_temp_files(self, day: dt.datetime):
        """
        Remove netcdf-files that has been created in the temp directory for the current month under the target directory.
        To be called if preprocessing of input or target data failed.
        :param day: date for which to remove temporary files
        """
        flist = glob.glob(os.path.join(self.target_dir, f"tmp_{day.strftime('%Y%m')}", "**", f"*{day.strftime('%Y%m%d')}.nc"), recursive=True)

        if flist:
            logger.debug(f"Found {len(flist)} files for {day.strftime('%Y-%m-%d')} that will be removed due to previous processing failure.")
            remove_files(list(flist))
        
    def get_grid_period(self, day):
        """
        Get grid period for a given day
        :param day: date for which to get grid period
        :return: grid period for given day
        """

        period_ends = [pd.Timestamp(period.split("-")[-1]) for period in self.grid_periods_all]

        start, end = pd.Timestamp(self.grid_periods_all[0].split("-")[0]), pd.Timestamp(self.grid_periods_all[-1].split("-")[-1])

        idx = 0
        if not (start <= day <= end):
            fmt = "%Y/%m/%d"                                                                             
            raise ValueError(f"Date must be between {start.strftime(fmt)} and {end.strftime(fmt)}.")

        for period_end in period_ends:
            if period_end >= day:
                break
            else: 
                idx += 1
        
        return self.grid_periods_all[idx]
    
    def setup_logger(self, log_file: str = None):
        """
        Set-up logger for preprocessing
        :param log_file: iabsolute (!) path to log file
        :return: logger object
        """
        # create file handler which logs even debug messages
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not log_file:
            log_file = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), "HPC_batch_scripts", 
                                    f"preprocess_euradim_{self.downscaling_fac}km_{current_time}.log")

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # add formatter to the handlers and add handlers to logger
        fh.setFormatter(formatter), ch.setFormatter(formatter)
        logger.addHandler(fh), logger.addHandler(ch)

    
    @staticmethod
    def get_cdo_string_nox(var_str: str):
        """
        Get CDO-string for chained operation to get NOx-data 
        :param var_str: comma-separated variable list as to be used with CDO-operator selname, must contain NOx
        :return: adapted CDO-string to get NOx 
        """
        if "NOx" not in var_str:
            raise ValueError(f"NOx is not part of query for variables ('{var_str}')")
        
        vars_aux = var_str.replace("NOx", "NO,NO2")

        cdo_str_new = f"-aexpr,'NOx=NO + NO2' -selname,{vars_aux}"

        return cdo_str_new  
