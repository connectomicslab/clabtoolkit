from pathlib import Path
from typing import Union
from typing import Union, Optional, Dict, List
import json


########################################################################################
def get_sidecars_files(nifti_path: Union[str, Path]) -> Dict[str, Optional[Path]]:
    """
    Get the DWI sidecar files (bvec, bval, json) for a given NIfTI file.

    Parameters
    ----------
    nifti_path : Union[str, Path]
        Path to the NIfTI file.

    Returns
    -------
    Dict[str, Optional[Path]]
        Dictionary containing paths to the bvec, bval, and json files.
    """

    if isinstance(nifti_path, str):
        nifti_path = Path(nifti_path)

    stem = nifti_path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parent = nifti_path.parent
    bvec = parent / f"{stem}.bvec"
    bval = parent / f"{stem}.bval"
    json_ = parent / f"{stem}.json"

    return {
        "bvec": bvec if bvec.exists() else None,
        "bval": bval if bval.exists() else None,
        "json": json_ if json_.exists() else None,
    }


########################################################################################
def merge_json_files(json_paths: List[Union[str, Path]]) -> Dict:
    """
    Merge multiple JSON sidecar dicts into one.

    - Fields identical across all files are kept as-is.
    - Fields that differ (or are absent in some files) use the value
        from the first file.

    Parameters
    ----------
    json_paths : List[Union[str, Path]]
        List of paths to the JSON sidecar files.

    Returns
    -------
    Dict
        Merged JSON dictionary.

    """
    loaded = []
    for p in json_paths:

        if isinstance(p, str):
            p = Path(p)

        with open(p, "r") as f:
            loaded.append(json.load(f))

    all_keys = set().union(*loaded)
    merged = {}

    for key in all_keys:
        values = [d.get(key, None) for d in loaded]
        # Keep scalar if all values are identical, otherwise fall back to first
        merged[key] = values[0]

    return merged
