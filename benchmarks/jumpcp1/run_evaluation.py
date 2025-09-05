import pandas as pd
import pipelines


def get_downstream_result(feature_extractor):
    replicate_feature = "Metadata_broad_sample"
    batch = "2020_11_04_CPJUMP1"
    experiment_df = (
        pd.read_csv("metadata/experiment-metadata.tsv", sep="\t")
        .query("Batch==@batch")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )
    plates_to_analyze = ['BR00117010', 'BR00117011', 'BR00117012', 'BR00117013', 'BR00117024', 'BR00117025', 'BR00117026'] #compound plates
    experiment_df = experiment_df[experiment_df["Assay_Plate_Barcode"].isin(plates_to_analyze)]

    target1_metadata = pd.read_csv(
        "metadata/JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        sep="\t",
        usecols=["broad_sample", "target_list"],
    ).rename(
        columns={
            "broad_sample": "Metadata_broad_sample",
            "target_list": "Metadata_target_list",
        }
    )

    replicability_map_df = pd.DataFrame()
    replicability_fr_df = pd.DataFrame()
    matching_map_df = pd.DataFrame()
    matching_fr_df = pd.DataFrame()

    if feature_extractor == 'cellprofiler':
        data_inputs = ['bygroupfilt_cellpaint_spherized_featsel_0.001']
    else:
        data_inputs = ['bygroupfilt_spherized_0.001']

    for data_input in data_inputs:
        well_level_data = pipelines.load_data(experiment_df, feature_extractor, data_input)
        replicability_map_df, replicability_fr_df = pipelines.replicability_pipeline(replicability_map_df, replicability_fr_df, well_level_data, data_input)
        matching_map_df, matching_fr_df = pipelines.matching_pipeline_compound(well_level_data, matching_map_df, matching_fr_df, replicability_map_df, replicate_feature, data_input, target1_metadata)

    replicability_fr_df.to_csv(f'./output/fr_replicability_{feature_extractor}_results.csv', index = False)
    replicability_map_df.to_csv(f'./output/map_replicability_{feature_extractor}_results.csv', index = False)
    matching_fr_df.to_csv(f'./output/fr_matching_{feature_extractor}_results.csv', index = False)
    matching_map_df.to_csv(f'./output/map_matching_{feature_extractor}_results.csv', index = False)


if __name__ == "__main__":
    feature_extractors = ['cpcnn']#, 'dino4cells','openphenom_comp_chmean', 'openphenom_comp_allch'] #cellprofiler evaluated 
    for feature_extractor in feature_extractors:
        get_downstream_result(feature_extractor)