# This code is reorganized from Chandrasekaran et al. 2024 Nature Methods paper
# Original repository: https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods/ 

import utils
import numpy as np
import pandas as pd
from typing import Optional

def load_data(timepoint_df: pd.DataFrame, 
              feature_extractor:str, 
              data_input:str):
    well_level_data = pd.DataFrame()
    for plate in timepoint_df.Assay_Plate_Barcode.unique():
    # Read all the plates
        data_df = utils.load_data(feature_extractor, plate, f"{data_input}.csv.gz")

        if data_input == 'spherized' and feature_extractor == 'cellprofiler':
            data_df = data_df[data_df.Metadata_Plate == plate].reset_index(drop=True)
        well_level_data = utils.concat_profiles(well_level_data, data_df)

    well_level_data.loc[well_level_data.Metadata_pert_iname == 'DMSO', 'Metadata_broad_sample'] = 'DMSO'

    well_level_data = utils.remove_empty_wells(well_level_data)
    well_level_data["Metadata_negcon"] = np.where(
        well_level_data["Metadata_control_type"] == "negcon", 1, 0
    )

    well_level_data.reset_index(inplace=True, drop=True)
    return well_level_data


def load_data_deeplearning(timepoint_df: pd.DataFrame, 
                           feature_extractor:str, 
                           data_input:str):
    
    well_level_data = pd.DataFrame()
    for plate in timepoint_df.Assay_Plate_Barcode.unique():
    # Read all the plates
        data_df = utils.load_data(feature_extractor, plate, f"{data_input}.csv.gz")
        well_level_data = utils.concat_profiles(well_level_data, data_df)

    well_level_data.loc[well_level_data.Metadata_pert_iname == 'DMSO', 'Metadata_broad_sample'] = 'DMSO'
    well_level_data = utils.remove_empty_wells(well_level_data)
    well_level_data["Metadata_negcon"] = np.where(
        well_level_data["Metadata_control_type"] == "negcon", 1, 0
    )
    well_level_data.reset_index(inplace=True, drop=True)
    return well_level_data


def replicability_pipeline(replicability_map_df:pd.DataFrame,
                            replicability_fr_df:pd.DataFrame,
                            well_level_data: pd.DataFrame,
                            data_input: str):
    batch_size = 100000
    null_size = 100000

    pos_sameby = ["Metadata_broad_sample"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = ["Metadata_negcon"]

    metadata_df = utils.get_metadata(well_level_data)
    feature_df = utils.get_featuredata(well_level_data)
    feature_values = feature_df.values

    result = utils.run_pipeline(
        metadata_df,
        feature_values,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        anti_match=False,
        batch_size=batch_size,
        null_size=null_size,
    )

    result = result.query("Metadata_negcon==0").reset_index(drop=True)

    replicability_map_df, replicability_fr_df = utils.create_replicability_df(
        replicability_map_df,
        replicability_fr_df,
        result,
        pos_sameby,
        0.05,
        data_input,
        len(well_level_data)
    )

    return replicability_map_df, replicability_fr_df


def matching_pipeline_compound(well_level_profiles:pd.DataFrame,
                               matching_map_df:pd.DataFrame,
                               matching_fr_df:pd.DataFrame,
                               replicability_map_df: pd.DataFrame, 
                               replicate_feature:str,
                               data_input: str,
                               target1_metadata:pd.DataFrame):

    batch_size = 100000
    null_size = 100000
    # Remove DMSO wells
    well_level_profiles = utils.remove_negcon_and_empty_wells(well_level_profiles)

    # Create consensus profiles
    consensus_profiles = utils.consensus(well_level_profiles, replicate_feature)

    # Filter out non-replicable compounds
    replicable_compounds = list(
        replicability_map_df[(replicability_map_df.above_q_threshold == True)][replicate_feature]
    )
    consensus_profiles = consensus_profiles.query(
        "Metadata_broad_sample==@replicable_compounds"
    ).reset_index(drop=True)

    # Adding additional gene annotation metadata
    consensus_profiles = (
        consensus_profiles.merge(
            target1_metadata, on="Metadata_broad_sample", how="left"
        )
        .assign(
            Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|")
        )
        .drop(["Metadata_target_list"], axis=1)
    )

    # Calculate compound-compound matching
    pos_sameby = ["Metadata_matching_target"]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = ["Metadata_matching_target"]

    metadata_df = utils.get_metadata(consensus_profiles)
    feature_df = utils.get_featuredata(consensus_profiles)
    feature_values = feature_df.values
    result = utils.run_pipeline(
        metadata_df,
        feature_values,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        anti_match=True,
        batch_size=batch_size,
        null_size=null_size,
        multilabel_col="Metadata_matching_target",
    )

    matching_map_df, matching_fr_df = utils.create_matching_df(
        matching_map_df,
        matching_fr_df,
        result,
        pos_sameby,
        0.05,
        data_input
    )

    return matching_map_df, matching_fr_df