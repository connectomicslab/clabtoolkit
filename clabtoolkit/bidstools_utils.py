# bidstools_utils.py
"""
Helper utilities supporting :mod:`clabtoolkit.bidstools`.

Currently hosts the simulator used to build synthetic BIDS datasets for
testing, demos and documentation examples.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover - tqdm is optional
    _HAS_TQDM = False


SUPPORTED_MODALITIES = ("anat", "func", "dwi", "fmap")


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Section 1: Methods dedicated to simulate BIDs datasets           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def create_a_simulated_bids_dataset(
    output_dir: str | Path,
    n_subjects: int = 5,
    n_visits: int = 1,
    modalities: Sequence[str] = ("anat", "func", "dwi"),
    *,
    tasks: Sequence[str] = ("rest",),
    runs_per_modality: int = 1,
    anat_suffixes: Sequence[str] = ("T1w",),
    image_shape: tuple[int, int, int] = (64, 64, 40),
    voxel_size: tuple[float, float, float] = (3.0, 3.0, 3.0),
    n_volumes_func: int = 10,
    repetition_time: float = 2.0,
    n_directions_dwi: int = 15,
    dataset_name: str = "SimulatedBIDSDataset",
    add_events_tsv: bool = True,
    include_derivatives: str | Sequence[str] | None = None,
    random_seed: int | None = 42,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Create a synthetic dataset following the BIDS folder/file conventions.

    Generates ``sub-XX[/ses-YY]/<modality>/`` directories filled with
    placeholder NIfTI (``.nii.gz``) images, matching JSON sidecars, and
    the top-level BIDS metadata files (``dataset_description.json``,
    ``participants.tsv``, ``README``, ``CHANGES``).

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Root directory in which the dataset will be created. Created if
        it does not already exist.
    n_subjects : int, default 5
        Number of subjects to generate (``sub-01`` ... ``sub-N``).
    n_visits : int, default 1
        Number of sessions (visits) per subject. If ``1``, no
        ``ses-XX`` level is created (flat subject layout), matching
        standard BIDS behavior for single-session studies.
    modalities : sequence of str, default ("anat", "func", "dwi")
        Which imaging modalities to simulate. Supported values are
        ``"anat"``, ``"func"``, ``"dwi"``, and ``"fmap"``.
    tasks : sequence of str, default ("rest",)
        Task names used to build ``task-<label>`` entities for
        functional (``func``) data. Ignored if ``"func"`` is not in
        ``modalities``.
    runs_per_modality : int, default 1
        Number of runs (``run-XX``) generated per modality/task
        combination. Use ``1`` to omit the ``run`` entity entirely.
    anat_suffixes : sequence of str, default ("T1w",)
        Anatomical image suffixes to generate, e.g. ``("T1w", "T2w")``.
    image_shape : tuple of int, default (64, 64, 40)
        Spatial (x, y, z) shape used for all generated volumes.
    voxel_size : tuple of float, default (3.0, 3.0, 3.0)
        Voxel size in mm, encoded in the NIfTI affine.
    n_volumes_func : int, default 10
        Number of timepoints (4th dimension) in simulated ``bold`` runs.
    repetition_time : float, default 2.0
        Repetition time (seconds) written into the ``bold`` JSON
        sidecars.
    n_directions_dwi : int, default 15
        Number of diffusion-weighted directions (non-b0 volumes)
        simulated for ``dwi`` runs. One additional b0 volume is
        prepended.
    dataset_name : str, default "SimulatedBIDSDataset"
        Value written to ``dataset_description.json``'s ``"Name"``
        field.
    add_events_tsv : bool, default True
        If True, write a dummy ``*_events.tsv`` alongside each
        functional run.
    include_derivatives : str, sequence of str, or None, default None
        Name(s) of derivative pipeline(s) to simulate, e.g.
        ``"fmriprep"`` or ``["fmriprep", "freesurfer"]``. For each
        name, a ``derivatives/<pipeline>/`` folder is created at the
        dataset root with its own ``dataset_description.json``
        (``DatasetType: "derivative"``) and a
        ``sub-XX[/ses-YY]/<modality>/`` tree that mirrors the raw
        dataset's subject/session/modality organization, per the
        BIDS-Derivatives convention. If None (default), no
        ``derivatives/`` folder is created.
    random_seed : int or None, default 42
        Seed for reproducible dummy image data and participant
        metadata. Use ``None`` for non-deterministic output.
    overwrite : bool, default False
        If False, raises ``FileExistsError`` when ``output_dir``
        already exists and is non-empty. If True, files are written
        into it regardless (existing unrelated files are left alone).
    show_progress : bool, default True
        Display a progress bar over subjects (requires ``tqdm``; falls
        back silently to no progress bar if unavailable).

    Returns
    -------
    pathlib.Path
        Path to the root of the created BIDS dataset (``output_dir``).

    Raises
    ------
    ValueError
        If an unsupported modality is requested, or numeric parameters
        are invalid.
    FileExistsError
        If ``output_dir`` already exists, is non-empty, and
        ``overwrite=False``.

    Examples
    --------
    >>> create_a_simulated_bids_dataset(
    ...     "/tmp/my_fake_bids",
    ...     n_subjects=3,
    ...     n_visits=2,
    ...     modalities=["anat", "func", "dwi"],
    ...     tasks=["rest", "nback"],
    ... )
    PosixPath('/tmp/my_fake_bids')
    """
    # ---- validation -----------------------------------------------------
    unknown = set(modalities) - set(SUPPORTED_MODALITIES)
    if unknown:
        raise ValueError(
            f"Unsupported modalities {sorted(unknown)}. "
            f"Supported: {SUPPORTED_MODALITIES}"
        )
    if n_subjects < 1 or n_visits < 1 or runs_per_modality < 1:
        raise ValueError(
            "n_subjects, n_visits, and runs_per_modality must all be >= 1."
        )
    derivative_names = _normalize_derivatives(include_derivatives)

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"'{output_dir}' already exists and is not empty. "
            "Pass overwrite=True to write into it anyway."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    affine = np.diag(list(voxel_size) + [1.0]).astype(float)

    subject_labels = [f"sub-{i:02d}" for i in range(1, n_subjects + 1)]
    session_labels: list[str | None] = (
        [None] if n_visits == 1 else [f"ses-{v:02d}" for v in range(1, n_visits + 1)]
    )

    # ---- top-level metadata ----------------------------------------------
    _write_dataset_description(output_dir, dataset_name)
    _write_readme(output_dir, dataset_name)
    _write_changes(output_dir)
    _write_participants_tsv(output_dir, subject_labels, rng)

    # ---- raw per-subject/session/modality generation -----------------------
    _generate_imaging_tree(
        root=output_dir,
        subject_labels=subject_labels,
        session_labels=session_labels,
        modalities=modalities,
        tasks=tasks,
        runs_per_modality=runs_per_modality,
        anat_suffixes=anat_suffixes,
        image_shape=image_shape,
        n_volumes_func=n_volumes_func,
        repetition_time=repetition_time,
        n_directions_dwi=n_directions_dwi,
        affine=affine,
        rng=rng,
        add_events_tsv=add_events_tsv,
        show_progress=show_progress,
        progress_desc="Simulating BIDS subjects",
    )

    # ---- derivatives ---------------------------------------------------------
    for pipeline in derivative_names:
        deriv_root = output_dir / "derivatives" / pipeline
        deriv_root.mkdir(parents=True, exist_ok=True)
        _write_dataset_description(
            deriv_root,
            name=f"{dataset_name} ({pipeline})",
            dataset_type="derivative",
            generated_by_name=pipeline,
            source_dataset_name=dataset_name,
        )
        _generate_imaging_tree(
            root=deriv_root,
            subject_labels=subject_labels,
            session_labels=session_labels,
            modalities=modalities,
            tasks=tasks,
            runs_per_modality=runs_per_modality,
            anat_suffixes=anat_suffixes,
            image_shape=image_shape,
            n_volumes_func=n_volumes_func,
            repetition_time=repetition_time,
            n_directions_dwi=n_directions_dwi,
            affine=affine,
            rng=rng,
            add_events_tsv=add_events_tsv,
            show_progress=show_progress,
            progress_desc=f"Simulating '{pipeline}' derivatives",
        )

    summary = (
        f"Simulated BIDS dataset created at: {output_dir}\n"
        f"  Subjects: {n_subjects} | Sessions/subject: {n_visits} | "
        f"Modalities: {list(modalities)}"
    )
    if derivative_names:
        summary += f"\n  Derivatives: {derivative_names}"
    print(summary)
    return output_dir


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                 Section 2: Robust I/O helper methods                       ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def _write_json(path: Path, data: dict) -> None:
    """
    Write ``data`` as JSON to ``path``, forcing the write to hit disk.

    Uses an explicit ``flush()`` + ``os.fsync()`` before closing the file
    handle. This avoids files that appear zero-byte / empty when
    inspected immediately after creation on network or parallel
    filesystems (e.g. NFS, Lustre, GPFS) commonly used on HPC clusters,
    where buffered writes can otherwise lag behind what ``ls`` reports.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    data : dict
        JSON-serializable content.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())


####################################################################################################
def _write_text(path: Path, text: str) -> None:
    """
    Write plain text to ``path``, forcing the write to hit disk.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    text : str
        Content to write.
    """
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############         Section 3: Derivatives support (normalization + tree)             ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def _normalize_derivatives(
    include_derivatives: str | Sequence[str] | None,
) -> list[str]:
    """
    Normalize ``include_derivatives`` into a de-duplicated list of names.

    Parameters
    ----------
    include_derivatives : str, sequence of str, or None
        A single pipeline name, a sequence of names, or None.

    Returns
    -------
    list of str
        Cleaned, order-preserving, de-duplicated pipeline names. Empty
        if ``include_derivatives`` is None.

    Raises
    ------
    ValueError
        If any entry is not a non-empty string.
    """
    if include_derivatives is None:
        return []

    names = (
        [include_derivatives]
        if isinstance(include_derivatives, str)
        else list(include_derivatives)
    )

    seen: set[str] = set()
    normalized: list[str] = []
    for raw_name in names:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                "Each entry in include_derivatives must be a non-empty string "
                f"(got {raw_name!r})."
            )
        cleaned = raw_name.strip()
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


####################################################################################################
def _generate_imaging_tree(
    root: Path,
    subject_labels: Sequence[str],
    session_labels: Sequence[str | None],
    modalities: Sequence[str],
    tasks: Sequence[str],
    runs_per_modality: int,
    anat_suffixes: Sequence[str],
    image_shape: tuple[int, int, int],
    n_volumes_func: int,
    repetition_time: float,
    n_directions_dwi: int,
    affine: np.ndarray,
    rng: np.random.Generator,
    add_events_tsv: bool,
    show_progress: bool,
    progress_desc: str,
) -> None:
    """
    Populate a ``sub-XX[/ses-YY]/<modality>/`` imaging tree under ``root``.

    Shared by the raw dataset and every requested derivatives pipeline,
    since BIDS-Derivatives datasets mirror the same subject/session/
    modality organization as the raw data they were generated from.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root to populate (either the raw dataset root, or a
        ``derivatives/<pipeline>/`` root).
    subject_labels : sequence of str
        Subject identifiers, e.g. ``["sub-01", "sub-02"]``.
    session_labels : sequence of (str or None)
        Session identifiers, e.g. ``["ses-01", "ses-02"]``, or
        ``[None]`` for a flat, session-less layout.
    modalities : sequence of str
        Modalities to generate per subject/session.
    tasks : sequence of str
        Task labels for functional data.
    runs_per_modality : int
        Number of runs generated per modality/task combination.
    anat_suffixes : sequence of str
        Anatomical suffixes to generate.
    image_shape : tuple of int
        Spatial (x, y, z) shape used for all generated volumes.
    n_volumes_func : int
        Number of timepoints in simulated ``bold`` runs.
    repetition_time : float
        Repetition time (seconds) for ``bold`` sidecars.
    n_directions_dwi : int
        Number of diffusion-weighted directions for ``dwi`` runs.
    affine : numpy.ndarray
        4x4 affine matrix stored in every generated image header.
    rng : numpy.random.Generator
        Random generator used to draw voxel intensities and metadata.
    add_events_tsv : bool
        If True, write a dummy ``*_events.tsv`` alongside each
        functional run.
    show_progress : bool
        Display a per-subject progress bar (requires ``tqdm``).
    progress_desc : str
        Label shown on the progress bar, distinguishing the raw pass
        from each derivatives pipeline pass.
    """
    iterator = subject_labels
    if show_progress and _HAS_TQDM:
        iterator = tqdm(subject_labels, desc=progress_desc)

    for sub in iterator:
        for ses in session_labels:
            entity_prefix = f"{sub}_{ses}" if ses else sub
            base_dir = root / sub / ses if ses else root / sub

            for modality in modalities:
                mod_dir = base_dir / modality
                mod_dir.mkdir(parents=True, exist_ok=True)

                if modality == "anat":
                    _simulate_anat(
                        mod_dir, entity_prefix, anat_suffixes, image_shape, affine, rng
                    )
                elif modality == "func":
                    _simulate_func(
                        mod_dir,
                        entity_prefix,
                        tasks,
                        runs_per_modality,
                        image_shape,
                        n_volumes_func,
                        repetition_time,
                        affine,
                        rng,
                        add_events_tsv,
                    )
                elif modality == "dwi":
                    _simulate_dwi(
                        mod_dir,
                        entity_prefix,
                        runs_per_modality,
                        image_shape,
                        n_directions_dwi,
                        affine,
                        rng,
                    )
                elif modality == "fmap":
                    _simulate_fmap(mod_dir, entity_prefix, image_shape, affine, rng)


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############             Section 4: Top-level BIDS metadata generators                  ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def _write_dataset_description(
    root: Path,
    name: str,
    dataset_type: str = "raw",
    generated_by_name: str | None = None,
    source_dataset_name: str | None = None,
) -> None:
    """
    Write the dataset-level ``dataset_description.json`` file.

    Used for both the raw dataset and every ``derivatives/<pipeline>/``
    sub-dataset, which per the BIDS-Derivatives convention must carry
    its own ``dataset_description.json``.

    Parameters
    ----------
    root : pathlib.Path
        Root of the dataset (raw root, or a derivatives pipeline root).
    name : str
        Value stored in the ``"Name"`` field.
    dataset_type : str, default "raw"
        Either ``"raw"`` or ``"derivative"``.
    generated_by_name : str or None, default None
        Name recorded under ``"GeneratedBy"``. Defaults to this
        function's own fully-qualified name when None.
    source_dataset_name : str or None, default None
        When ``dataset_type == "derivative"``, the name of the raw
        dataset this derivative was generated from, recorded under
        ``"SourceDatasets"``.
    """
    description = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": dataset_type,
        "GeneratedBy": [
            {
                "Name": generated_by_name
                or "clabtoolkit.bidstools_utils.create_a_simulated_bids_dataset"
            }
        ],
    }
    if dataset_type == "derivative" and source_dataset_name is not None:
        description["SourceDatasets"] = [{"Name": source_dataset_name}]
    _write_json(root / "dataset_description.json", description)


####################################################################################################
def _write_readme(root: Path, name: str) -> None:
    """
    Write the dataset ``README`` file.

    Parameters
    ----------
    root : pathlib.Path
        Root of the simulated BIDS dataset.
    name : str
        Dataset name used as the README title.
    """
    _write_text(
        root / "README",
        f"# {name}\n\n"
        "This is a synthetic dataset generated for testing purposes. "
        "All imaging data are randomly generated placeholders and do "
        "not represent real acquisitions.\n",
    )


####################################################################################################
def _write_changes(root: Path) -> None:
    """
    Write the dataset ``CHANGES`` file.

    Parameters
    ----------
    root : pathlib.Path
        Root of the simulated BIDS dataset.
    """
    _write_text(root / "CHANGES", "1.0.0 - Initial simulated release\n")


####################################################################################################
def _write_participants_tsv(
    root: Path, subject_labels: Sequence[str], rng: np.random.Generator
) -> None:
    """
    Write ``participants.tsv`` and its JSON sidecar.

    Parameters
    ----------
    root : pathlib.Path
        Root of the simulated BIDS dataset.
    subject_labels : sequence of str
        Participant identifiers (e.g. ``["sub-01", "sub-02"]``).
    rng : numpy.random.Generator
        Random generator used to draw the dummy demographics.
    """
    lines = ["participant_id\tage\tsex"]
    for sub in subject_labels:
        age = int(rng.integers(18, 65))
        sex = rng.choice(["M", "F"])
        lines.append(f"{sub}\t{age}\t{sex}")
    _write_text(root / "participants.tsv", "\n".join(lines) + "\n")

    sidecar = {
        "age": {"Description": "Age of participant", "Units": "years"},
        "sex": {
            "Description": "Sex of participant",
            "Levels": {"M": "Male", "F": "Female"},
        },
    }
    _write_json(root / "participants.json", sidecar)


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                Section 5: Per-modality data simulators                     ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def _save_nifti(
    path: Path, shape: tuple[int, ...], affine: np.ndarray, rng: np.random.Generator
) -> None:
    """
    Save a NIfTI image filled with Gaussian noise.

    Parameters
    ----------
    path : pathlib.Path
        Destination ``.nii.gz`` file.
    shape : tuple of int
        Shape of the array to generate.
    affine : numpy.ndarray
        4x4 affine matrix stored in the image header.
    rng : numpy.random.Generator
        Random generator used to draw the voxel intensities.
    """
    data = rng.normal(loc=500, scale=50, size=shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


####################################################################################################
def _run_entity(run_idx: int, n_runs: int) -> str:
    """
    Build the ``run`` entity string, omitted for single-run acquisitions.

    Parameters
    ----------
    run_idx : int
        1-based run index.
    n_runs : int
        Total number of runs generated for the modality.

    Returns
    -------
    str
        ``"_run-XX"`` when ``n_runs > 1``, otherwise an empty string.
    """
    return f"_run-{run_idx:02d}" if n_runs > 1 else ""


####################################################################################################
def _simulate_anat(
    mod_dir: Path,
    entity_prefix: str,
    anat_suffixes: Sequence[str],
    image_shape: tuple[int, int, int],
    affine: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Simulate the anatomical (``anat``) images of a subject/session.

    Parameters
    ----------
    mod_dir : pathlib.Path
        Destination ``anat`` folder.
    entity_prefix : str
        Filename prefix, e.g. ``"sub-01_ses-01"``.
    anat_suffixes : sequence of str
        Suffixes to generate, e.g. ``("T1w", "T2w")``.
    image_shape : tuple of int
        Spatial (x, y, z) shape of the generated volumes.
    affine : numpy.ndarray
        4x4 affine matrix stored in the image headers.
    rng : numpy.random.Generator
        Random generator used to draw the voxel intensities.
    """
    for suffix in anat_suffixes:
        stem = f"{entity_prefix}_{suffix}"
        _save_nifti(mod_dir / f"{stem}.nii.gz", image_shape, affine, rng)
        sidecar = {
            "Modality": "MR",
            "MagneticFieldStrength": 3,
            "Manufacturer": "SimulatedScanner",
            "ScanningSequence": "GR",
        }
        _write_json(mod_dir / f"{stem}.json", sidecar)


####################################################################################################
def _simulate_func(
    mod_dir: Path,
    entity_prefix: str,
    tasks: Sequence[str],
    n_runs: int,
    image_shape: tuple[int, int, int],
    n_volumes: int,
    tr: float,
    affine: np.ndarray,
    rng: np.random.Generator,
    add_events_tsv: bool,
) -> None:
    """
    Simulate the functional (``func``) runs of a subject/session.

    Parameters
    ----------
    mod_dir : pathlib.Path
        Destination ``func`` folder.
    entity_prefix : str
        Filename prefix, e.g. ``"sub-01_ses-01"``.
    tasks : sequence of str
        Task labels used to build the ``task-<label>`` entity.
    n_runs : int
        Number of runs generated per task.
    image_shape : tuple of int
        Spatial (x, y, z) shape of the generated volumes.
    n_volumes : int
        Number of timepoints in each ``bold`` series.
    tr : float
        Repetition time, in seconds.
    affine : numpy.ndarray
        4x4 affine matrix stored in the image headers.
    rng : numpy.random.Generator
        Random generator used to draw the voxel intensities and events.
    add_events_tsv : bool
        If True, write a dummy ``*_events.tsv`` per run.
    """
    for task in tasks:
        for run_idx in range(1, n_runs + 1):
            run_ent = _run_entity(run_idx, n_runs)
            stem = f"{entity_prefix}_task-{task}{run_ent}_bold"
            full_shape = image_shape + (n_volumes,)
            _save_nifti(mod_dir / f"{stem}.nii.gz", full_shape, affine, rng)

            sidecar = {
                "RepetitionTime": tr,
                "EchoTime": 0.03,
                "TaskName": task,
                "PhaseEncodingDirection": "j-",
                "SliceTiming": [
                    round(i * tr / image_shape[2], 4) for i in range(image_shape[2])
                ],
            }
            _write_json(mod_dir / f"{stem}.json", sidecar)

            if add_events_tsv:
                n_events = max(1, n_volumes // 3)
                lines = ["onset\tduration\ttrial_type"]
                onset = 0.0
                for _ in range(n_events):
                    duration = round(float(rng.uniform(1.0, 3.0)), 2)
                    trial_type = rng.choice(["stimulus", "rest"])
                    lines.append(f"{round(onset, 2)}\t{duration}\t{trial_type}")
                    onset += duration + float(rng.uniform(0.5, 2.0))
                events_stem = f"{entity_prefix}_task-{task}{run_ent}_events"
                _write_text(mod_dir / f"{events_stem}.tsv", "\n".join(lines) + "\n")


####################################################################################################
def _simulate_dwi(
    mod_dir: Path,
    entity_prefix: str,
    n_runs: int,
    image_shape: tuple[int, int, int],
    n_directions: int,
    affine: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Simulate the diffusion (``dwi``) runs of a subject/session.

    Each run contains one b0 volume followed by ``n_directions``
    diffusion-weighted volumes, with matching ``.bval`` and ``.bvec``
    files.

    Parameters
    ----------
    mod_dir : pathlib.Path
        Destination ``dwi`` folder.
    entity_prefix : str
        Filename prefix, e.g. ``"sub-01_ses-01"``.
    n_runs : int
        Number of runs to generate.
    image_shape : tuple of int
        Spatial (x, y, z) shape of the generated volumes.
    n_directions : int
        Number of diffusion-weighted directions (non-b0 volumes).
    affine : numpy.ndarray
        4x4 affine matrix stored in the image headers.
    rng : numpy.random.Generator
        Random generator used to draw the voxel intensities and bvecs.
    """
    n_volumes = n_directions + 1  # +1 b0 volume
    for run_idx in range(1, n_runs + 1):
        run_ent = _run_entity(run_idx, n_runs)
        stem = f"{entity_prefix}{run_ent}_dwi"
        full_shape = image_shape + (n_volumes,)
        _save_nifti(mod_dir / f"{stem}.nii.gz", full_shape, affine, rng)

        bvals = [0] + [1000] * n_directions
        bvecs = rng.normal(size=(3, n_volumes))
        bvecs[:, 0] = 0.0  # b0 direction is null
        norms = np.linalg.norm(bvecs, axis=0)
        norms[norms == 0] = 1.0
        bvecs = bvecs / norms

        _write_text(mod_dir / f"{stem}.bval", " ".join(str(b) for b in bvals) + "\n")
        bvec_text = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in bvecs) + "\n"
        _write_text(mod_dir / f"{stem}.bvec", bvec_text)

        sidecar = {
            "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.05,
            "EchoTime": 0.09,
        }
        _write_json(mod_dir / f"{stem}.json", sidecar)


####################################################################################################
def _simulate_fmap(
    mod_dir: Path,
    entity_prefix: str,
    image_shape: tuple[int, int, int],
    affine: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Simulate the field-map (``fmap``) images of a subject/session.

    Generates a phase-difference field map: two magnitude images plus
    the ``phasediff`` image and its sidecar.

    Parameters
    ----------
    mod_dir : pathlib.Path
        Destination ``fmap`` folder.
    entity_prefix : str
        Filename prefix, e.g. ``"sub-01_ses-01"``.
    image_shape : tuple of int
        Spatial (x, y, z) shape of the generated volumes.
    affine : numpy.ndarray
        4x4 affine matrix stored in the image headers.
    rng : numpy.random.Generator
        Random generator used to draw the voxel intensities.
    """
    for suffix in ("magnitude1", "magnitude2", "phasediff"):
        stem = f"{entity_prefix}_{suffix}"
        _save_nifti(mod_dir / f"{stem}.nii.gz", image_shape, affine, rng)

    sidecar = {
        "EchoTime1": 0.00519,
        "EchoTime2": 0.00765,
        "IntendedFor": [],
    }
    _write_json(mod_dir / f"{entity_prefix}_phasediff.json", sidecar)
