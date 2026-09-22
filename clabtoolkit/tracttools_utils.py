"""
Utility functions supporting the tractogram simulation of :mod:`clabtoolkit.tracttools`.

This module gathers the geometric primitives used to build synthetic tractograms:
the extraction of the voxel-grid geometry of a reference image, a
rotation-minimizing frame along a curve, smooth random deviation profiles, the
generation of a bundle centroid, the growth of a streamline population around it
and the noise that turns the resulting ideal geometry into a noisy tractogram.
They are kept apart from ``tracttools`` so the main module stays focused on
loading, manipulating and saving tractograms.

These functions work on plain numpy arrays and do not depend on the
``Tractogram`` class, so they can also be reused to build other kinds of
synthetic curves.

Examples
--------
>>> import numpy as np
>>> import clabtoolkit.tracttools_utils as tractutils
>>> rng = np.random.default_rng(42)
>>> _, _, _, bbox_min, bbox_max = tractutils.get_reference_geometry('fa.nii.gz')
>>> centroid = tractutils.simulate_centroid(rng, bbox_min, bbox_max, (50, 80))
>>> bundle = tractutils.populate_bundle(rng, centroid, n_streamlines=100, radius=4.0)
"""

import os
from pathlib import Path
from typing import Union

import nibabel as nb
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

###############################################################################################
def get_reference_geometry(
    ref_image: Union[str, Path, nb.Nifti1Image],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the voxel-grid geometry of a reference image.

    Parameters
    ----------
    ref_image : str, Path or nibabel image
        Reference NIfTI image (or a path to it) defining the space in which the
        streamlines will be simulated.

    Returns
    -------
    affine : np.ndarray
        4x4 voxel-to-RASmm affine matrix of the reference image.

    dims : np.ndarray
        Image dimensions (3,).

    zooms : np.ndarray
        Voxel sizes in mm (3,).

    bbox_min : np.ndarray
        Lower corner of the image bounding box, in world (RAS mm) coordinates.

    bbox_max : np.ndarray
        Upper corner of the image bounding box, in world (RAS mm) coordinates.
    """

    if isinstance(ref_image, (str, Path)):
        ref_image = str(ref_image)
        if not os.path.isfile(ref_image):
            raise FileNotFoundError(f"Reference image not found: {ref_image}")
        img = nb.load(ref_image)
    elif hasattr(ref_image, "affine") and hasattr(ref_image, "shape"):
        img = ref_image
    else:
        raise TypeError(
            "ref_image must be a path to a NIfTI file or a nibabel image object"
        )

    if len(img.shape) < 3:
        raise ValueError("The reference image must have at least 3 dimensions")

    affine = np.asarray(img.affine, dtype=float)
    dims = np.array(img.shape[:3], dtype=int)
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)

    # The 8 corners of the voxel grid. Voxel centers span -0.5 to n - 0.5 in
    # voxel units, so this covers the full field of view of the image.
    corners_vox = np.array(
        [
            [x, y, z]
            for x in (-0.5, dims[0] - 0.5)
            for y in (-0.5, dims[1] - 0.5)
            for z in (-0.5, dims[2] - 0.5)
        ],
        dtype=float,
    )
    corners_world = nb.affines.apply_affine(affine, corners_vox)

    return affine, dims, zooms, corners_world.min(axis=0), corners_world.max(axis=0)


###############################################################################################
def parallel_transport_frame(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a rotation-minimizing orthonormal frame along a 3D curve.

    The Frenet frame flips whenever the curvature vanishes, which produces
    twisting artifacts when it is used to place streamlines around a centroid.
    This function propagates the normal vector by parallel transport instead,
    yielding a frame that varies smoothly along the whole curve.

    Parameters
    ----------
    points : np.ndarray
        Curve coordinates with shape (n_points, 3).

    Returns
    -------
    tangents : np.ndarray
        Unit tangent vectors, shape (n_points, 3).

    normals : np.ndarray
        Unit normal vectors, shape (n_points, 3).

    binormals : np.ndarray
        Unit binormal vectors, shape (n_points, 3).
    """

    points = np.asarray(points, dtype=float)

    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms

    # Seed the frame with any unit vector that is not parallel to the first tangent
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, tangents[0]))) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])

    normals = np.zeros_like(tangents)
    first_normal = seed - np.dot(seed, tangents[0]) * tangents[0]
    normals[0] = first_normal / np.linalg.norm(first_normal)

    for i in range(1, len(tangents)):
        axis = np.cross(tangents[i - 1], tangents[i])
        sin_angle = np.linalg.norm(axis)

        if sin_angle < 1e-8:
            # The tangent did not change, so the normal is carried over unchanged
            normals[i] = normals[i - 1]
        else:
            axis = axis / sin_angle
            angle = np.arctan2(sin_angle, float(np.dot(tangents[i - 1], tangents[i])))
            prev = normals[i - 1]
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Rodrigues rotation of the previous normal onto the new tangent plane
            normals[i] = (
                prev * cos_a
                + np.cross(axis, prev) * sin_a
                + axis * float(np.dot(axis, prev)) * (1.0 - cos_a)
            )

        normals[i] = normals[i] / np.linalg.norm(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals


###############################################################################################
def smooth_random_deviation(
    rng: np.random.Generator,
    n_points: int,
    n_control: int,
    amplitude: float,
    n_dim: int = 2,
    fix_ends: bool = True,
) -> np.ndarray:
    """
    Builds a smooth, low-frequency random deviation profile.

    A few random control values are drawn and interpolated with a cubic spline,
    so the resulting profile wanders gently instead of jittering point by point.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used to draw the control values.

    n_points : int
        Number of samples of the returned profile.

    n_control : int
        Number of random control values. Fewer control points produce smoother
        and longer-wavelength deviations.

    amplitude : float
        Standard deviation (in mm) of the random control values.

    n_dim : int, optional
        Dimensionality of the profile. Default is 2 (the plane perpendicular
        to the centroid).

    fix_ends : bool, optional
        Whether to force the deviation to be zero at both extremities.
        Default is True.

    Returns
    -------
    np.ndarray
        Deviation profile with shape (n_points, n_dim).
    """

    n_control = max(int(n_control), 2)

    if amplitude <= 0:
        return np.zeros((n_points, n_dim))

    control = rng.normal(0.0, amplitude, size=(n_control, n_dim))
    if fix_ends:
        control[0] = 0.0
        control[-1] = 0.0

    t_control = np.linspace(0.0, 1.0, n_control)
    t_eval = np.linspace(0.0, 1.0, n_points)

    return CubicSpline(t_control, control, axis=0)(t_eval)


###############################################################################################
def simulate_centroid(
    rng: np.random.Generator,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    length_range: tuple[float, float],
    curvature: float = 0.05,
    n_points: int = 100,
    margin: float = 0.12,
    direction: np.ndarray = None,
    endpoints: tuple[np.ndarray, np.ndarray] = None,
) -> np.ndarray:
    """
    Generates a single, gently curved streamline used as the centroid of a bundle.

    The centroid is built as a straight segment whose interior control points are
    displaced perpendicular to its main axis, and then interpolated with a cubic
    spline. The displacement is proportional to the length of the segment, so
    `curvature` behaves as a scale-free "bending" parameter.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    bbox_min, bbox_max : np.ndarray
        Bounding box of the reference space, in world (RAS mm) coordinates.

    length_range : tuple of float
        Minimum and maximum centroid length in mm.

    curvature : float, optional
        Bending of the centroid, as a fraction of its length. Values around
        0.02-0.08 produce gently curved, anatomically plausible bundles, while
        larger values produce strongly bent ones. Default is 0.05.

    n_points : int, optional
        Number of points of the returned centroid. Default is 100.

    margin : float, optional
        Fraction of the bounding box kept free at each border, so the bundles do
        not touch the edges of the field of view. Default is 0.12.

    direction : np.ndarray, optional
        Main orientation of the centroid as a 3-element vector. If None, a random
        orientation is drawn. Ignored when `endpoints` is provided.

    endpoints : tuple of np.ndarray, optional
        Explicit start and end points, in world coordinates. When provided, the
        bounding box, the length range and the direction are not used.

    Returns
    -------
    np.ndarray
        Centroid coordinates with shape (n_points, 3), in world (RAS mm) coordinates.
    """

    if endpoints is not None:
        start, end = np.asarray(endpoints[0], dtype=float), np.asarray(
            endpoints[1], dtype=float
        )
        half = float(np.linalg.norm(end - start)) / 2.0

    else:
        inner_min = bbox_min + margin * (bbox_max - bbox_min)
        inner_max = bbox_max - margin * (bbox_max - bbox_min)
        extents = inner_max - inner_min

        if direction is None:
            direction = rng.normal(size=3)
        direction = np.asarray(direction, dtype=float)

        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("The direction vector cannot be null")
        direction = direction / norm

        half = float(rng.uniform(*length_range)) / 2.0

        # Longest half-length that this orientation can accommodate inside the box
        half_max = np.inf
        for axis in range(3):
            if abs(direction[axis]) > 1e-8:
                half_max = min(half_max, (extents[axis] / 2.0) / abs(direction[axis]))
        half = min(half, half_max)

        # The center is drawn only where the whole segment is guaranteed to fit
        low = inner_min + half * np.abs(direction)
        high = inner_max - half * np.abs(direction)
        center = low + rng.random(3) * np.maximum(high - low, 0.0)

        start, end = center - half * direction, center + half * direction

    # Straight backbone, later displaced perpendicular to itself
    n_control = 5
    t_control = np.linspace(0.0, 1.0, n_control)
    control = start + np.outer(t_control, end - start)

    if curvature > 0 and half > 0:
        _, normals, binormals = parallel_transport_frame(control)
        lateral = smooth_random_deviation(
            rng, n_control, n_control, curvature * 2.0 * half, n_dim=2
        )
        control = control + lateral[:, [0]] * normals + lateral[:, [1]] * binormals

    # Chord-length parameterization keeps the spacing of the points regular
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(control, axis=0), axis=1))]
    if chord[-1] <= 0:
        raise ValueError(
            "The simulated centroid has zero length. Check the reference image "
            "geometry and the 'bundle_length' parameter."
        )
    chord = chord / chord[-1]

    return CubicSpline(chord, control, axis=0)(np.linspace(0.0, 1.0, n_points))


###############################################################################################
def populate_bundle(
    rng: np.random.Generator,
    centroid: np.ndarray,
    n_streamlines: int,
    radius: float,
    spread: float = 0.35,
    fanning: float = 0.3,
    length_variability: float = 0.08,
    n_points: int = 100,
) -> list[np.ndarray]:
    """
    Builds a population of streamlines around a centroid.

    Each streamline is a copy of the centroid displaced in the plane perpendicular
    to it. The displacement has a constant component (which sets the position of
    the streamline inside the cross-section of the bundle) and a smooth random
    component (which makes the streamlines wander around the centroid instead of
    being exactly parallel to it).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    centroid : np.ndarray
        Centroid coordinates with shape (n_centroid_points, 3).

    n_streamlines : int
        Number of streamlines of the bundle.

    radius : float
        Radius of the cross-section of the bundle in mm.

    spread : float, optional
        Amplitude of the random wandering of each streamline, as a fraction of
        `radius`. Default is 0.35.

    fanning : float, optional
        Relative widening of the bundle towards its extremities. A value of 0
        produces a perfect tube. Default is 0.3.

    length_variability : float, optional
        Maximum fraction of the centroid trimmed at each extremity, so the
        streamlines of the bundle do not all end at the same place.
        Default is 0.08.

    n_points : int, optional
        Number of points of every simulated streamline. Default is 100.

    Returns
    -------
    list of np.ndarray
        The simulated streamlines, each one with shape (n_points, 3).
    """

    centroid = np.asarray(centroid, dtype=float)
    _, normals, binormals = parallel_transport_frame(centroid)

    t_centroid = np.linspace(0.0, 1.0, len(centroid))

    # Bundles are slightly wider at their extremities than at their core
    taper = 1.0 + fanning * (2.0 * np.abs(t_centroid - 0.5)) ** 2

    streamlines = []
    for _ in range(int(n_streamlines)):

        # Uniform sampling of the disk defining the cross-section of the bundle
        rho = radius * np.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * np.pi)
        offset = np.array([rho * np.cos(theta), rho * np.sin(theta)])

        offsets = np.outer(taper, offset) + smooth_random_deviation(
            rng, len(centroid), 4, spread * radius, n_dim=2
        )

        streamline = (
            centroid + offsets[:, [0]] * normals + offsets[:, [1]] * binormals
        )

        # Trimming the extremities so the streamlines have different lengths
        if length_variability > 0:
            t_start = rng.uniform(0.0, length_variability)
            t_end = 1.0 - rng.uniform(0.0, length_variability)
            t_new = np.linspace(t_start, t_end, n_points)
            streamline = CubicSpline(t_centroid, streamline, axis=0)(t_new)

        elif len(streamline) != n_points:
            t_new = np.linspace(0.0, 1.0, n_points)
            streamline = CubicSpline(t_centroid, streamline, axis=0)(t_new)

        streamlines.append(streamline.astype(np.float32))

    return streamlines


###############################################################################################
def add_noise_to_streamlines(
    rng: np.random.Generator,
    streamlines: Union[list[np.ndarray], np.ndarray],
    noise_level: float,
    noise_smoothness: float = 0.0,
    fix_endpoints: bool = False,
) -> Union[list[np.ndarray], np.ndarray]:
    """
    Adds a random displacement to every point of one or several streamlines.

    The noise is isotropic in the three directions of the world space and its
    amplitude is expressed in millimetres, so it can be related to the voxel size
    of the acquisition. Two regimes are available. With `noise_smoothness` equal
    to 0 every point is displaced independently, which produces the jagged,
    high-frequency aspect of streamlines reconstructed from noisy diffusion data.
    With a positive `noise_smoothness` the displacement is correlated along the
    streamline, which produces a slowly wandering trajectory instead. In both
    cases the amplitude is rescaled after the smoothing, so `noise_level` always
    corresponds to the standard deviation actually applied.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    streamlines : list of np.ndarray, ArraySequence or np.ndarray
        Streamlines to perturb. A single streamline can be passed as an array of
        shape (n_points, 3), in which case a single array is returned.

    noise_level : float
        Standard deviation of the displacement in mm. Values of 0 or less leave
        the streamlines untouched.

    noise_smoothness : float, optional
        Correlation length of the noise along the streamline, as a fraction of
        its number of points. 0 produces independent noise at every point
        (rough streamlines), while values around 0.05-0.2 produce smooth
        deviations from the original trajectory. Must be in the range [0, 1].
        Default is 0.

    fix_endpoints : bool, optional
        Whether to leave the first and the last point of each streamline
        unchanged. Useful when the extremities must stay inside a mask or a
        region of interest. Default is False.

    Returns
    -------
    list of np.ndarray or np.ndarray
        The perturbed streamlines. The return type matches the input: a single
        array in, a single array out.

    Raises
    ------
    ValueError
        If `noise_smoothness` is outside the range [0, 1].

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> noisy = add_noise_to_streamlines(rng, streamlines, noise_level=0.8)

    >>> # Smooth deviations instead of point-by-point roughness
    >>> wavy = add_noise_to_streamlines(rng, streamlines, noise_level=2.0,
    ...                                 noise_smoothness=0.1)
    """

    if not 0 <= noise_smoothness <= 1:
        raise ValueError(
            f"noise_smoothness must be in the range [0, 1], got {noise_smoothness}"
        )

    single_input = isinstance(streamlines, np.ndarray) and streamlines.ndim == 2
    if single_input:
        streamlines = [streamlines]

    if noise_level <= 0:
        return streamlines[0] if single_input else list(streamlines)

    noisy_streamlines = []
    for streamline in streamlines:
        streamline = np.asarray(streamline, dtype=float)
        n_points = len(streamline)

        if n_points == 0:
            noisy_streamlines.append(streamline.astype(np.float32))
            continue

        noise = rng.normal(0.0, 1.0, size=(n_points, 3))

        if noise_smoothness > 0:
            sigma = noise_smoothness * n_points
            noise = gaussian_filter1d(noise, sigma=sigma, axis=0, mode="nearest")

            # Smoothing shrinks the variance, so the amplitude is restored here
            # to keep noise_level meaningful whatever the correlation length.
            # The root mean square is used rather than the standard deviation:
            # a strongly smoothed field has a non-zero mean, and normalizing by
            # its standard deviation would overshoot the requested amplitude.
            current_rms = float(np.sqrt(np.mean(noise**2)))
            if current_rms > 1e-12:
                noise = noise / current_rms

        noise = noise * noise_level

        if fix_endpoints and n_points > 1:
            noise[0] = 0.0
            noise[-1] = 0.0

        noisy_streamlines.append((streamline + noise).astype(np.float32))

    return noisy_streamlines[0] if single_input else noisy_streamlines


###############################################################################################
def simulate_noise_streamlines(
    rng: np.random.Generator,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    n_streamlines: int,
    length_range: tuple[float, float],
    curvature: float = 0.15,
    n_points: int = 100,
    margin: float = 0.12,
    mask_coords: np.ndarray = None,
) -> list[np.ndarray]:
    """
    Simulates isolated streamlines that do not belong to any bundle.

    Real tractograms contain spurious streamlines that no bundle claims: broken
    or wandering trajectories produced by the tracking algorithm. This function
    generates them as independent centroids, each one with its own random
    orientation, a length drawn from a wider range and a higher curvature than
    the bundles, so they stand out as outliers rather than as a coherent
    population.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    bbox_min, bbox_max : np.ndarray
        Bounding box of the reference space, in world (RAS mm) coordinates.

    n_streamlines : int
        Number of spurious streamlines to generate.

    length_range : tuple of float
        Length range of the bundles in mm. The spurious streamlines are drawn
        from a wider interval, going down to a third of the minimum length, so
        short fragments are also produced.

    curvature : float, optional
        Bending of the spurious streamlines, as a fraction of their length.
        Higher than the bundle default on purpose. Default is 0.15.

    n_points : int, optional
        Number of points of every simulated streamline. Default is 100.

    margin : float, optional
        Fraction of the bounding box kept free at each border. Default is 0.12.

    mask_coords : np.ndarray, optional
        World coordinates of the voxels of a mask. When provided, the
        extremities of the spurious streamlines are drawn inside it.

    Returns
    -------
    list of np.ndarray
        The spurious streamlines, each one with shape (n_points, 3).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> outliers = simulate_noise_streamlines(rng, bbox_min, bbox_max, 50, (50, 90))
    """

    wide_range = (max(length_range[0] / 3.0, 1e-3), length_range[1])

    streamlines = []
    for _ in range(int(n_streamlines)):

        endpoints = None
        if mask_coords is not None:
            endpoints = sample_endpoints_in_mask(rng, mask_coords, wide_range)

        streamline = simulate_centroid(
            rng,
            bbox_min,
            bbox_max,
            wide_range,
            curvature=curvature,
            n_points=n_points,
            margin=margin,
            endpoints=endpoints,
        )
        streamlines.append(streamline.astype(np.float32))

    return streamlines


###############################################################################################
def mask_world_coordinates(
    mask: Union[str, Path, np.ndarray, nb.Nifti1Image],
    affine: np.ndarray,
    max_points: int = 50000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Returns the world coordinates of the non-zero voxels of a mask.

    Parameters
    ----------
    mask : str, Path, np.ndarray or nibabel image
        Mask restricting where the bundles can be placed. A path or a nibabel
        image is read with its own affine, while a bare array is assumed to live
        on the grid described by `affine`.

    affine : np.ndarray
        Voxel-to-RASmm affine of the reference image, used when the mask is
        supplied as a plain array.

    max_points : int, optional
        Maximum number of voxels kept. Larger masks are randomly subsampled to
        keep the endpoint search fast. Default is 50000.

    rng : np.random.Generator, optional
        Random number generator used for the subsampling.

    Returns
    -------
    np.ndarray
        World coordinates of the mask voxels, with shape (n_voxels, 3).
    """

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(mask, (str, Path)):
        mask = str(mask)
        if not os.path.isfile(mask):
            raise FileNotFoundError(f"Mask image not found: {mask}")
        mask_img = nb.load(mask)
        mask_data = mask_img.get_fdata()
        mask_affine = np.asarray(mask_img.affine, dtype=float)

    elif isinstance(mask, np.ndarray):
        mask_data = mask
        mask_affine = np.asarray(affine, dtype=float)

    elif hasattr(mask, "affine") and hasattr(mask, "get_fdata"):
        mask_data = mask.get_fdata()
        mask_affine = np.asarray(mask.affine, dtype=float)

    else:
        raise TypeError(
            "mask must be a path, a numpy array or a nibabel image object"
        )

    voxels = np.argwhere(mask_data > 0)
    if voxels.size == 0:
        raise ValueError("The supplied mask does not contain any non-zero voxel")

    if len(voxels) > max_points:
        voxels = voxels[rng.choice(len(voxels), size=max_points, replace=False)]

    return nb.affines.apply_affine(mask_affine, voxels.astype(float))



###############################################################################################
def sample_endpoints_in_mask(
    rng: np.random.Generator,
    mask_coords: np.ndarray,
    length_range: tuple[float, float],
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draws a pair of mask voxels separated by a distance inside `length_range`.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    mask_coords : np.ndarray
        World coordinates of the mask voxels, with shape (n_voxels, 3).

    length_range : tuple of float
        Minimum and maximum distance in mm between the two returned points.

    max_attempts : int, optional
        Number of starting points tried before falling back to the pair that is
        closest to the requested range. Default is 200.

    Returns
    -------
    tuple of np.ndarray
        The two selected points, in world coordinates.
    """

    min_len, max_len = length_range
    best_pair, best_error = None, np.inf

    for _ in range(max_attempts):
        start = mask_coords[rng.integers(len(mask_coords))]
        distances = np.linalg.norm(mask_coords - start, axis=1)

        candidates = np.flatnonzero((distances >= min_len) & (distances <= max_len))
        if candidates.size:
            return start, mask_coords[rng.choice(candidates)]

        # Keeping track of the closest match in case no exact one is found
        errors = np.maximum(min_len - distances, distances - max_len)
        idx = int(np.argmin(errors))
        if errors[idx] < best_error:
            best_error, best_pair = errors[idx], (start, mask_coords[idx])

    return best_pair
