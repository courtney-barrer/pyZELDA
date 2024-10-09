
import numpy as np
import pylab as pl
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import matplotlib.pyplot as plt
from pathlib import Path 

from astropy.io import fits


def insert_array(big_array, small_array):
    """
    Inserts a smaller array into the center of a larger array, 
    handling both even and odd dimensions correctly.
    
    Parameters:
    - big_array: The larger array of size (N, M).
    - small_array: The smaller array of size (n, m).
    
    Returns:
    - big_array: The larger array with the smaller array inserted at its center.
    """
    N, M = big_array.shape
    n, m = small_array.shape

    # Calculate the starting indices to place the smaller array in the center
    start_x = (N - n) // 2
    start_y = (M - m) // 2
    
    # Handle even/odd cases by adjusting the end index correctly
    end_x = start_x + n
    end_y = start_y + m
    
    # Insert the smaller array into the center of the big array
    big_array[start_x:end_x, start_y:end_y] = small_array

    return big_array


def crop_pupil(pupil, image):
    """
    Detects the boundary of a pupil in a binary mask (with pupil = 1 and background = 0)
    and crops both the pupil mask and the corresponding image to contain just the pupil.
    
    Parameters:
    - pupil: A 2D NumPy array (binary) representing the pupil (1 inside the pupil, 0 outside).
    - image: A 2D NumPy array of the same shape as 'pupil' representing the image to be cropped.
    
    Returns:
    - cropped_pupil: The cropped pupil mask.
    - cropped_image: The cropped image based on the pupil's bounding box.
    """
    # Ensure both arrays have the same shape
    if pupil.shape != image.shape:
        raise ValueError("Pupil and image must have the same dimensions.")

    # Sum along the rows (axis=1) to find the non-zero rows (pupil region)
    row_sums = np.sum(pupil, axis=1)
    non_zero_rows = np.where(row_sums > 0)[0]

    # Sum along the columns (axis=0) to find the non-zero columns (pupil region)
    col_sums = np.sum(pupil, axis=0)
    non_zero_cols = np.where(col_sums > 0)[0]

    # Get the bounding box of the pupil by identifying the min and max indices
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1] + 1
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1] + 1

    # Crop both the pupil and the image
    cropped_pupil = pupil[row_start:row_end, col_start:col_end]
    cropped_image = image[row_start:row_end, col_start:col_end]

    return cropped_pupil, cropped_image



# Sensor definition (instrument is NEAR, clear aperture for internal source)
z = zelda.Sensor('BALDR_UT_J3')

# check our cauchy fit of the refractive index is good! 
#wvl_grid = np.linspace(1e-6,1.7e-6,100)
#plt.plot( wvl_grid, [ztools.refractive_index(wave = w, substrate = 'N_1405') for w in wvl_grid] ) ;plt.show()

pupil_basis = ztools.zernike.zernike_basis(nterms=15, npix=z.pupil_diameter, rho=None, theta=None) * 1e-9


opd_input = z.pupil * insert_array(np.zeros( z.pupil.shape), 0 * np.nan_to_num( pupil_basis[4] ) )

amp = z.pupil #* insert_array(np.zeros( z.pupil.shape), 100*np.nan_to_num( abs(pupil_basis[1] ) ) )

N0 = ztools.propagate_opd_map(opd_map= 0*opd_input, mask_diameter = z._mask_diameter, mask_depth = 0*z.mask_depth, mask_substrate = z.mask_substrate, mask_Fratio=z._Fratio,
                      pupil_diameter=z.pupil_diameter, pupil = amp**0.5 , wave = 1.65e-6)

I0 = ztools.propagate_opd_map(opd_map= 0*opd_input, mask_diameter = z._mask_diameter, mask_depth = z.mask_depth, mask_substrate = z.mask_substrate, mask_Fratio=z._Fratio,
                      pupil_diameter=z.pupil_diameter, pupil = amp**0.5 , wave = 1.65e-6)

I = ztools.propagate_opd_map(opd_map= opd_input, mask_diameter = z._mask_diameter, mask_depth = z.mask_depth, mask_substrate = z.mask_substrate, mask_Fratio=z._Fratio,
                      pupil_diameter=z.pupil_diameter, pupil = amp**0.5 , wave = 1.65e-6)

#plt.figure()
#plt.imshow( crop_pupil( z.pupil, I )[0] ); plt.show()


# Standard analysis
z_opd_standard = z.analyze(clear_pupil=N0, zelda_pupil=I, wave=1.25e-6)

# Advanced analysis
pupil_roi = aperture.disc(z.pupil_diameter, z.pupil_diameter, diameter=True, cpix=False)
z_opd_advanced = z.analyze(clear_pupil=N0, zelda_pupil=I, wave=1.25e-6,
                           use_arbitrary_amplitude=True,
                           refwave_from_clear=True,
                           cpix=False, pupil_roi=z.pupil)



reference_wave, expi = ztools.create_reference_wave_beyond_pupil(z.mask_diameter, 
                                                                    z.mask_depth,
                                                                    z.mask_substrate,
                                                                    z.mask_Fratio,
                                                                    z.pupil.shape[1],
                                                                    z.pupil,
                                                                    1.65e-6)



fig = plt.figure(0, figsize=(24, 4))
plt.clf()

gs = gridspec.GridSpec(ncols=8, nrows=1, figure=fig, width_ratios=[.1,1,1,1,1,1,1,.1])

ax = fig.add_subplot(gs[0,1])
mappable = ax.imshow(N0, aspect='equal', vmin=0, vmax=1)
ax.set_title('Clear pupil')

ax = fig.add_subplot(gs[0,0])
cbar1 = fig.colorbar(mappable=mappable, cax=ax)
cbar1.set_label('Normalized intensity')

ax = fig.add_subplot(gs[0,2])
ax.imshow(I, aspect='equal', vmin=0, vmax=1)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(gs[0,3])
ax.imshow(1e9 * opd_input, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Introduced aberration (nm)')

ax = fig.add_subplot(gs[0,4])
cax = ax.imshow(z_opd_standard, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Reconstructed aberration - standard')

ax = fig.add_subplot(gs[0,5])
cax = ax.imshow(z_opd_advanced, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Reconstructed aberration - advanced')

ax = fig.add_subplot(gs[0,6])
cax = ax.imshow(1e9 * opd_input - z_opd_advanced, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Residual')

ax = fig.add_subplot(gs[0,7])
cbar = fig.colorbar(mappable=cax, cax=ax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()
