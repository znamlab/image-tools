from typing import Optional

import numpy as np
import numpy.typing as npt

from . import phase_correlation as _pc


def phase_correlation_by_block(
    reference: npt.NDArray,
    target: npt.NDArray,
    block_size: int = 256,
    overlap: float = 0.1,
):
    """Estimate translation between two images by dividing them into blocks and 
    estimating translation for each block.

    Args:
        reference (np.array): reference image
        target (np.array): target image
        block_size (int, optional): size of the blocks, defaults to 256
        overlap (float, optional): fraction of overlap between blocks, defaults to 0.1

    Returns:
        shifts (np.array): array of shifts for each block
        corrs (np.array): array of correlation coefficients for each block
        block_centers (np.array): array of centers of each block

    """
    step_size = int(block_size * (1 - overlap))
    nblocks = np.array(reference.shape) // step_size
    block_centers = np.zeros((*nblocks, 2))
    shifts = np.zeros((*nblocks, 2))
    corrs = np.zeros(nblocks)
    for row in range(nblocks[0]):
        for col in range(nblocks[1]):
            ref_block = reference[
                row * step_size : row * step_size + block_size,
                col * step_size : col * step_size + block_size,
            ]
            target_block = target[
                row * step_size : row * step_size + block_size,
                col * step_size : col * step_size + block_size,
            ]
            shift, corr = _pc.phase_correlation(
                ref_block,
                target_block,
            )[:2]
            shifts[row, col] = shift
            corrs[row, col] = corr
            block_centers[row, col] = np.array(
                [row * step_size + block_size // 2, col * step_size + block_size // 2]
            )
    return shifts, corrs, block_centers
