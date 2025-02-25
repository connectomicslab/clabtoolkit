#### Testing the grouping by codes

# import clabtoolkit.parcellationtools as cltparc
# import clabtoolkit.misctools as cltmisc
# import numpy as np

# self = cltparc.Parcellation(parc_file='/media/COSAS/Yasser/Work2Do/TestChimera/derivatives/freesurfer/sub-CHUVA001_ses-V2_run-01_acq-mpragep3/tmp/aparc+aseg.nii.gz')

# self._load_colortable('/opt/freesurfer/FreeSurferColorLUT.txt')

# self._group_by_code([['1000:1999'],['2000:2999']],[1,2])


# self._remove_by_name(names2remove=['Thalamus'])

# self._save_parcellation('/home/yaleman/cer.nii.gz',save_lut=True)
# a = 1

#### Testing the entitiy changes
# import clabtoolkit.bidstools as cltbids
# import os
# from bids import BIDSLayout
# from tqdm import tqdm

# bids_dir = '/media/HPCdata/Sandra/MCIC/'
# project = 'MCIC'

# # Detecting the subject folders
# layout = BIDSLayout(bids_dir, validate=False, derivatives= False)
# subj_ids = layout.get(return_type='id', target='subject', suffix='T1w')


# for subj_id in tqdm(subj_ids):
#     subj_dir = os.path.join(bids_dir, 'sub-'+subj_id)
#     new_subj_id = project + subj_id

#     cltbids._recursively_replace_entity_value(subj_dir,dict2old= {'sub':subj_id}, dict2new ={'sub':new_subj_id})

#     # rename the subject folder
#     os.rename(subj_dir, os.path.join(bids_dir, 'sub-'+new_subj_id))
#     a = 1


#### Testing the crop_image_from_mask

# import clabtoolkit.imagetools as cltimg


# in_image = '/home/yaleman/.cache/templateflow/tpl-SUIT/tpl-SUIT_atlas-Diedrichsen2009_probseg.nii.gz'

# mask = '/home/yaleman/.cache/templateflow/tpl-SUIT/tpl-SUIT_atlas-Diedrichsen2009_dseg.nii.gz'
# cropped_image = '/media/COSAS/Yasser/Work2Do/TestChimera/test_crooped.nii.gz'
# st_codes = ["1-5"]
# cltimg.crop_image_from_mask(in_image,
#                             mask,
#                             cropped_image,
#                             st_codes)
# restored= '/media/COSAS/Yasser/Work2Do/TestChimera/test_restored.nii.gz'
# cltimg.cropped_to_native(cropped_image,
#                             in_image,
#                             restored)


# import copy
# import clabtoolkit.parcellationtools as cltparc
# import clabtoolkit.misctools as cltmisc

# parc = cltparc.Parcellation(
#     "/home/yaleman/sub-ctrl10_ses-1_rec-DEN_atlas-chimeraLFMIHIFIFN_scale-1_desc-grow1mm_dseg.nii.gz"
# )
# indexes = cltmisc.get_indexes_by_substring(
#     list1=parc.name, substr=["thal-", "hipp-", "hipp-", "amygd-", "subc-"]
# )

# # Overwritting the original parcellation object
# parc.group_by_code(indexes + 1, new_codes=1)
# parc.save_parcellation("/home/yaleman/Desktop/parc.nii.gz", save_lut=True)

# # If you don't want to overwrite the original parcellation object
# tmpparc = copy.deepcopy(parc)
# tmpparc.group_by_code(indexes + 1, new_codes=1)
# tmpparc.save_parcellation("/home/yaleman/Desktop/parc.nii.gz", save_lut=True)


# # If you don't want to overwrite the original parcellation object
# parc.group_by_name(["thal-", "hipp-", "hipp-", "amygd-", "subc-"], new_codes=1)
# parc.save_parcellation("/home/yaleman/Desktop/parc_byname.nii.gz", save_lut=True)


# # If you don't want to overwrite the original parcellation object
# tmpparc = copy.deepcopy(parc)
# tmpparc.group_by_name(["thal-", "hipp-", "hipp-", "amygd-", "subc-"], new_codes=1)
# tmpparc.save_parcellation("/home/yaleman/Desktop/parc_byname.nii.gz", save_lut=True)


import clabtoolkit.freesurfertools as cltfree

annot_file = "/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot"

# annot_obj = cltfree.AnnotParcellation(parc_file=annot_file)
# annot_obj.group_into_lobes(
#     grouping="desikan",
#     lobes_json="/media/COSAS/Yasser/COSAS/GithubRepositories/Worktools/clabtoolkit/clabtoolkit/files/lobes.json",
#     out_annot="/opt/freesurfer/subjects/fsaverage/label/lh.lobes.annot",)


import os
import shutil
from typing import Union, Dict, List
import copy
from pyvista import _vtk, PolyData
from numpy import split, ndarray

import pandas as pd
import nibabel as nib
import numpy as np

import clabtoolkit.misctools as cltmisc
import clabtoolkit.freesurfertools as cltfree
import clabtoolkit.morphometrytools as cltmorphtools
import clabtoolkit.surfacetools as cltsurf
import clabtoolkit.parcellationtools as cltparc
import clabtoolkit.bidstools as cltbids


def create_subject_morpho_table(t1, chim_file, fssubj_dir, metrics, stats_list):

    # Detecting the base directory
    if not os.path.isfile(t1):
        raise ValueError("Please provide a valid T1 image.")

    # Getting the entities from the name
    anat_dir = os.path.dirname(t1)
    t1_name = os.path.basename(t1)
    ent_dict = cltbids.str2entity(t1_name)

    temp_entities = t1_name.split("_")[:-1]
    fullid = "_".join(temp_entities)
    ent_dict_fullid = cltbids.str2entity(fullid)

    if "ses" in ent_dict.keys():
        path_cad = "sub-" + ent_dict["sub"] + os.path.sep + "ses-" + ent_dict["ses"]
    else:
        path_cad = "sub-" + ent_dict["sub"]

    # Getting the freesurfer directory
    hemi = "lh"
    sub2proc = cltfree.FreeSurferSubject(fullid, subjs_dir=fssubj_dir)

    parc_file = metric_file = sub2proc.fs_files["surf"][hemi]["desikan"]
    metric_file = sub2proc.fs_files["surf"]["lh"]["thickness"]
    surf_file = sub2proc.fs_files["surf"]["lh"]["white"]

    lobar_obj = cltfree.AnnotParcellation(parc_file=parc_file)
    lobar_obj.group_into_lobes(grouping="desikan")

    # Thickness left hemisphere
    metric_file = sub2proc.fs_files["surf"][hemi]["thickness"]
    df_lobes_thick, metric_vect = cltmorphtools.compute_reg_val_fromannot(
        metric_file,
        lobar_obj,
        hemi,
        "thickness",
        stats_list,
    )

    df_region_thick, metric_vect = cltmorphtools.compute_reg_val_fromannot(
        metric_file, parc_file, hemi, "thickness", stats_list
    )

    # Merge the two dataframes
    df_thick = pd.concat([df_lobes_thick, df_region_thick], axis=0)

    df_region_area = cltmorphtools.compute_reg_area_fromsurf(
        surf_file, parc_file, hemi, include_unknown=False
    )
    df_lobe_area = cltmorphtools.compute_reg_area_fromsurf(
        surf_file, lobar_obj, hemi, include_unknown=True
    )

    df_area = pd.concat([df_lobe_area, df_region_area], axis=0)

    df_euler = cltmorphtools.compute_euler_fromsurf(surf_file, hemi)

    df_volume = cltmorphtools.compute_reg_volume_fromparcellation(parc_file=chim_file)

    # Concatenate the two dataframes
    df_all = pd.concat([df_thick, df_area, df_euler, df_volume], axis=0)

    all_entities = [
        "sub",
        "ses",
        "acq",
        "dir",
        "run",
        "ce",
        "rec",
        "space",
        "res",
        "model",
        "desc",
        "atlas",
        "scale",
        "seg",
        "grow",
    ]

    for entity in reversed(all_entities):

        if entity in ent_dict.keys():
            value = ent_dict[entity]
        else:
            value = ""

        if entity == "sub":
            df_all.insert(0, "participant_id", value)
        elif entity == "ses":
            df_all.insert(0, "session_id", value)
        elif entity == "atlas":
            if "chimera" in value:
                df_all.insert(0, "atlas_id", "chimera")
                # Remove the word chimera from tmp string
                tmp = value.replace("chimera", "")
                df_all.insert(1, "chimera_id", tmp)
            else:
                df_all.insert(0, "atlas_id", value)
                df_all.insert(1, "chimera_id", "")

        elif entity == "desc":
            df_all.insert(0, "desc_id", value)
            if "grow" in value:
                tmp = value.replace("grow", "")
                df_all.insert(1, "grow", tmp)
        else:
            df_all.insert(0, entity + "_id", value)

    return df_all


t1 = "/media/yaleman/HagmannHDD/IMAGING_PROJECTS/CLM/CLM/BIDS/sub-P0003038C002/ses-PRAD0786386/anat/sub-P0003038C002_ses-PRAD0786386_run-1_T1w.nii.gz"
chim_file = "/media/yaleman/HagmannHDD/IMAGING_PROJECTS/CLM/CLM/derivatives/chimera/sub-P0003038C002/ses-PRAD0786386/anat/sub-P0003038C002_ses-PRAD0786386_run-1_atlas-chimeraLFIIHISIFN_scale-1_desc-grow1mm_dseg.nii.gz"

fssubj_dir = (
    "/media/yaleman/HagmannHDD/IMAGING_PROJECTS/CLM/CLM/derivatives/freesurfer/"
)
metrics = "thickness"
stats_list = ["mean", "median", "std", "min", "max"]
tab = create_subject_morpho_table(t1, chim_file, fssubj_dir, metrics, stats_list)


parc_file = "/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot"
surf_file = "/opt/freesurfer/subjects/fsaverage/surf/lh.white"
lobe_var = "/opt/freesurfer/subjects/fsaverage/label/lh.aparc.lobes.annot"
metric_file = "/opt/freesurfer/subjects/fsaverage/surf/lh.thickness"
hemi = "lh"
stats_list = ["mean", "median", "std", "min", "max"]
# # # # # metric_vect = nib.freesurfer.io.read_morph_data(metric_file)

# # # # json_file = "/media/COSAS/Yasser/COSAS/GithubRepositories/Worktools/clabtoolkit/clabtoolkit/files/metrics_units.json"
# # # # import json

# # # # with open(json_file) as f:
#     metric_dict = json.load(f)


# annot_obj = cltfree.AnnotParcellation(parc_file = parc_file)
# lobe_obj = cltfree.AnnotParcellation(parc_file = lobe_var)

df_lobes_thick, metric_vect = cltmorphtools.compute_reg_val_fromannot(
    metric_file,
    lobe_var,
    hemi,
    "thickness",
    stats_list,
)
df_region_thick, metric_vect = cltmorphtools.compute_reg_val_fromannot(
    metric_file, parc_file, hemi, "thickness", stats_list
)

# Merge the two dataframes
df_thick = pd.concat([df_lobes_thick, df_region_thick], axis=0)

# Remove repeated columns (Mainly for the case of the hemisphere)
# Remove the rows with repeated regions


# # # # print(df_thick.head())

df_region_area = cltmorphtools.compute_reg_area_fromsurf(
    surf_file, parc_file, hemi, include_unknown=False
)
df_lobe_area = cltmorphtools.compute_reg_area_fromsurf(
    surf_file, lobe_var, hemi, include_unknown=False
)


df_area = pd.concat([df_lobe_area, df_region_area], axis=0)


df_euler = cltmorphtools.compute_euler_fromsurf(surf_file, hemi)

# Concatenate the two dataframes
df_all = pd.concat([df_thick, df_area, df_euler], axis=0)
print(df_all.head())


metric_file = "/media/HPCdata/Data/derivatives/dwi-preprocessing/sub-CHUVA013/ses-V3/dwi/odfreconst/sub-CHUVA013_ses-V3_run-01_acq-dsiNdir129_space-dwi_res-1x1x1_model-SHORE_desc-dipygfa_GFA.nii.gz"
parc_file = "/media/HPCdata/Data/derivatives/dwi-preprocessing/sub-CHUVA013/ses-V3/anat/parcellations/sub-CHUVA013_ses-V3_run-01_space-dwi_atlas-chimeraLRMIHIFIFN_scale-1_desc-grow2mm_dseg.nii.gz"
df_region_map = cltmorphtools.compute_reg_val_fromparcellation(
    metric_file, parc_file, "fa", stats_list
)

df_region_vol = cltmorphtools.compute_reg_volume_fromparcellation(parc_file=parc_file)


stat_file = "/opt/freesurfer/subjects/bert/stats/aseg.stats"

etiv = cltfree.parse_freesurfer_statsfile(stat_file)
a = 1

# Concate the dataframes
df_all = pd.concat([df_area, df_region_map, df_region_vol, etiv], axis=0)

# Save the dataframe as csv
df_all.to_csv("/media/HPCdata/Data/derivatives/df_all.csv")
