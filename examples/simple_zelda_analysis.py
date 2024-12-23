import matplotlib.pyplot as plt
import numpy as np

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture

from pathlib import Path

#%% parameters

# path = Path('/Users/mndiaye/Dropbox/python/zelda/pyZELDA/')
path = Path('/Users/avigan/Work/GitHub/pyZELDA/data/')
# path = Path('D:/Programmes/GitHub/pyZELDA/')

wave = 1.642e-6

# internal data
# clear_pupil_files = ['SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3', 'SPHERE_CLEAR_PUPIL_CUBE1_NDIT=3']
# zelda_pupil_files = ['SPHERE_ZELDA_PUPIL_CUBE1_NDIT=3', 'SPHERE_ZELDA_PUPIL_CUBE2_NDIT=3']

# dark_file  = 'SPHERE_BACKGROUND'
# pupil_tel  = False

# on-sky data
clear_pupil_files = ['SPHERE_GEN_IRDIS057_0002']
zelda_pupil_files = ['SPHERE_GEN_IRDIS057_0001']
dark_file  = 'SPHERE_GEN_IRDIS057_0003'
pupil_tel  = True

#%% ZELDA analysis
z = zelda.Sensor('SPHERE-IRDIS', pupil_telescope=pupil_tel)

clear_pupil, zelda_pupil, center = z.read_files(path, clear_pupil_files, zelda_pupil_files, dark_file,
                                                collapse_clear=True, collapse_zelda=False)

opd_map = z.analyze(clear_pupil, zelda_pupil, wave=wave)
if opd_map.ndim == 3:
    opd_map = opd_map.mean(axis=0)

# decomposition on Zernike polynomials
basis, coeff, opd_zern = ztools.zernike_expand(opd_map, 100)

opd_map_reconstructed = opd_zern.mean(axis=0)
opd_map_reconstructed[opd_map == 0] = np.nan

#%% plot
fig = plt.figure(0, figsize=(16, 4))
plt.clf()

ax = fig.add_subplot(141)
ax.imshow(clear_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=2000)
ax.set_title('Clear pupil')

ax = fig.add_subplot(142)
ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=2000)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(143)
ax.imshow(opd_map, aspect='equal', vmin=-150, vmax=150, cmap='magma')
ax.set_title('OPD map')

ax = fig.add_subplot(144)
cax = ax.imshow(opd_zern.mean(axis=0), aspect='equal', vmin=-150, vmax=150, cmap='magma')
ax.set_title('Zernike projected OPD map')

cbar = fig.colorbar(cax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()

#%%
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

ext = 100

fig = plt.figure('Intensity', figsize=(10, 8))
fig.clf()
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.06], bottom=0.05, top=0.95, left=0.02, right=0.88, wspace=0.1)
ax = fig.add_subplot(gs[0])
cim = ax.imshow(zelda_pupil.mean(axis=0), aspect='equal', vmin=0, vmax=2000, cmap='viridis')
ax.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_ticklabels([])
ax.yaxis.set_major_locator(ticker.NullLocator())
ax = fig.add_subplot(gs[1])
cbar = fig.colorbar(cim, cax=ax, orientation='vertical', label='Intensity [ADU]')
fig.savefig(path / 'zelda_intensity.pdf')

fig = plt.figure('OPD', figsize=(10, 8))
fig.clf()
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.06], bottom=0.05, top=0.95, left=0.02, right=0.88, wspace=0.1)
ax = fig.add_subplot(gs[0])
cim = ax.imshow(opd_map, aspect='equal', vmin=-ext, vmax=ext, cmap='bwr')
ax.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_ticklabels([])
ax.yaxis.set_major_locator(ticker.NullLocator())
ax = fig.add_subplot(gs[1])
cbar = fig.colorbar(cim, cax=ax, orientation='vertical', label='Optical path difference [nm]')
fig.savefig(path / 'zelda_opd.pdf')

fig = plt.figure('OPD reconstructed', figsize=(10, 8))
fig.clf()
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.06], bottom=0.05, top=0.95, left=0.02, right=0.88, wspace=0.1)
ax = fig.add_subplot(gs[0])
cim = ax.imshow(opd_map_reconstructed, aspect='equal', vmin=-ext, vmax=ext, cmap='bwr')
ax.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_ticklabels([])
ax.yaxis.set_major_locator(ticker.NullLocator())
ax = fig.add_subplot(gs[1])
cbar = fig.colorbar(cim, cax=ax, orientation='vertical', label='Optical path difference [nm]')
fig.savefig(path / 'zelda_opd_reconstructed.pdf')
