# -*- coding: utf-8 -*-
'''
pyZELDA main module

arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

import os
import sys
import numpy as np

import pyzelda.utils.mft as mft
import pyzelda.utils.imutils as imutils
import pyzelda.utils.aperture as aperture
import pyzelda.utils.circle_fit as circle_fit
import pyzelda.ztools as ztools

import poppy.zernike as zernike
import scipy.ndimage as ndimage

from astropy.io import fits

if sys.version_info < (3, 0):
    import ConfigParser
else:
    import configparser as ConfigParser


class Sensor():
    '''
    Zernike wavefront sensor class
    '''

    ##################################################
    # Constructor
    ##################################################
    
    def __init__(self, instrument):
        '''
        Initialization of the Sensor class

        Parameters
        ----------
        instrument : str
            Instrument associated with the sensor
        '''

        self._instrument = instrument

        # read configuration file
        package_directory = os.path.dirname(os.path.abspath(__file__))
        configfile = os.path.join(package_directory, 'instruments', instrument+'.ini')
        config = ConfigParser.ConfigParser()

        try:
            config.read(configfile)

            # mask physical parameters
            self._mask_depth = float(config.get('mask', 'depth'))
            self._mask_diameter = float(config.get('mask', 'diameter'))
            self._mask_substrate = config.get('mask', 'substrate')

            # instrument parameters
            self._pupil_diameter = int(config.get('instrument', 'pupil_diameter'))
            self._Fratio = float(config.get('instrument', 'Fratio'))

            # detector sub-window parameters
            self._width = int(config.get('detector_crop', 'width'))
            self._height = int(config.get('detector_crop', 'width'))
            cx = int(config.get('detector_crop', 'origin_x'))
            cy = int(config.get('detector_crop', 'origin_y'))
            self._origin = (cx, cy)
        except ConfigParser.Error as e:
            raise ValueError('Error reading {0} configuration file: {1}'.format(instrument, e.message))
        
    
    ##################################################
    # Properties
    ##################################################
    
    @property
    def instrument(self):
        return self._instrument

    @property
    def mask_depth(self):
        return self._mask_depth

    @property
    def mask_diameter(self):
        return self._mask_diameter

    @property
    def mask_substrate(self):
        return self._mask_substrate

    @property
    def Fratio(self):
        return self._Fratio

    @property
    def pupil_diameter(self):
        return self._pupil_diameter

    @property
    def detector_subwindow_width(self):
        return self._width

    @property
    def detector_subwindow_height(self):
        return self._height

    @property
    def detector_subwindow_origin(self):
        return self._origin

    ##################################################
    # Methods
    ##################################################
    
    def read_files(self, path, clear_pupil_files, zelda_pupil_files, dark_files, center=(), center_method='fit',
                   collapse_clear=False, collapse_zelda=False):
        '''
        Read a sequence of ZELDA files from disk and prepare them for analysis

        Parameters
        ----------
        path : str
            Path to the directory that contains the TIFF files

        clear_pupil_files : str
            List of files that contains the clear pupil data, without the .fits

        zelda_pupil_files : str
            List of files that contains the ZELDA pupil data, without the .fits

        dark_files : str
            List of files that contains the dark data, without the .fits

        center : tuple, optional
            Specify the center of the pupil in raw data coordinations.
            Default is '()', i.e. the center will be determined by the routine

        center_method : str, optional
            Method to be used for finding the center of the pupil:
             - 'fit': least squares circle fit (default)
             - 'com': center of mass

        collapse_clear : bool
            Collapse the clear pupil images. Default is False

        collapse_zelda : bool
            Collapse the zelda pupil images. Default is False

        Returns
        -------
        clear_pupil : array_like
            Array containing the collapsed clear pupil data

        zelda_pupil : array_like
            Array containing the zelda pupil data

        c : vector_like
            Vector containing the (x,y) coordinates of the center in 1024x1024 raw data format
        '''

        ##############################
        # Deal with files
        ##############################

        # read number of frames
        nframes_clear = ztools.number_of_frames(path, clear_pupil_files)	
        nframes_zelda = ztools.number_of_frames(path, zelda_pupil_files)	

        print('Clear pupil: nframes={0}, collapse={1}'.format(nframes_clear, collapse_clear))
        print('ZELDA pupil: nframes={0}, collapse={1}'.format(nframes_zelda, collapse_zelda))

        # make sure we have compatible data sets
        if (nframes_zelda == 1) or collapse_zelda:
            if nframes_clear != 1:
                collapse_clear = True
                print(' * automatic collapse of clear pupil to match ZELDA data')
        else:
            if (nframes_zelda != nframes_clear) and (not collapse_clear) and (nframes_clear != 1):
                raise ValueError('Incompatible number of frames between ZELDA and clear pupil. ' +
                                 'You could use collapse_clear=True.')

        # read dark data	
        dark = ztools.load_data(path, dark_files, self._width, self._height, self._origin)
        dark = dark.mean(axis=0)

        # read clear pupil data
        clear_pupil = ztools.load_data(path, clear_pupil_files, self._width, self._height, self._origin)

        ##############################
        # Center determination
        ##############################

        # collapse clear pupil image
        clear_pupil_collapse = clear_pupil.mean(axis=0, keepdims=True)

        # subtract background and correct for bad pixels
        clear_pupil_collapse -= dark
        clear_pupil_collapse = imutils.sigma_filter(clear_pupil_collapse.squeeze(), box=5, nsigma=3, iterate=True)

        # search for the pupil center
        if len(center) == 0:
            center = ztools.pupil_center(clear_pupil_collapse, center_method)
        elif len(center) != 2:
            raise ValueError('Error, you must pass 2 values for center')

        ##############################
        # Clean and recenter images
        ##############################
        clear_pupil = ztools.recentred_data_cubes(path, clear_pupil_files, dark, self._pupil_diameter,
                                                  center, collapse_clear, self._origin)
        zelda_pupil = ztools.recentred_data_cubes(path, zelda_pupil_files, dark, self._pupil_diameter,
                                                  center, collapse_zelda, self._origin)

        return clear_pupil, zelda_pupil, center
    

    def analyze(self, clear_pupil, zelda_pupil, wave, overwrite=False, silent=False):
        '''Performs the ZELDA data analysis using the outputs provided by the read_files() function.

        Parameters
        ----------
        clear_pupil : array_like
            Array containing the clear pupil data

        zelda_pupil : array_like
            Array containing the zelda pupil data

        wave : float, optional
            Wavelength of the data, in m.

        overwrite : bool
            If set to True, the OPD maps are saved inside the zelda_pupil
            array to save memory. Otherwise, a distinct OPD array is
            returned. Do not use if you're not a ZELDA High Master :-)

        silent : bool, optional
            Remain silent during the data analysis

        Returns
        -------
        opd : array_like
            Optical path difference map in nanometers

        '''

        #make sure we have 3D cubes
        if clear_pupil.ndim == 2:
            clear_pupil = clear_pupil[np.newaxis, ...]

        if zelda_pupil.ndim == 2:
            zelda_pupil = zelda_pupil[np.newaxis, ...]

        # create a copy of the zelda pupil array if needed
        if not overwrite:
            zelda_pupil = zelda_pupil.copy()

        # make sure wave is an array
        if type(wave) is not list:
            wave = [wave]
        wave  = np.array(wave)
        nwave = wave.size

        # ++++++++++++++++++++++++++++++++++
        # Geometrical parameters
        # ++++++++++++++++++++++++++++++++++
        pupil_diameter = self._pupil_diameter
        R_pupil_pixels = pupil_diameter/2

        # ++++++++++++++++++++++++++++++++++
        # Reference wave(s)
        # ++++++++++++++++++++++++++++++++++
        mask_diffraction_prop = []
        for w in wave:
            reference_wave, expi = ztools.create_reference_wave(self._mask_diameter, self._mask_depth,
                                                                self._mask_substrate,
                                                                pupil_diameter, self._Fratio, w)
            mask_diffraction_prop.append((reference_wave, expi))

        # ++++++++++++++++++++++++++++++++++
        # Phase reconstruction from data
        # ++++++++++++++++++++++++++++++++++
        pup = aperture.disc(pupil_diameter, R_pupil_pixels, mask=True, cpix=True, strict=True)

        print('ZELDA analysis')
        nframes_clear = len(clear_pupil)
        nframes_zelda = len(zelda_pupil)

        # (nframes_clear, nframes_zelda) is either (1, N) or (N, N). (N, 1) is not allowed.
        if (nframes_clear != nframes_zelda) and (nframes_clear != 1):
            raise ValueError('Incompatible number of frames between clear and ZELDA pupil images')

        if (nwave != 1) and (nwave != nframes_zelda):
            raise ValueError('Incompatible number of wavelengths and ZELDA pupil images')

        for idx in range(nframes_zelda):
            print(' * frame {0} / {1}'.format(idx+1, nframes_zelda))

            # normalization
            if nframes_clear == 1:
                zelda_norm = zelda_pupil[idx] / clear_pupil
            else:
                zelda_norm = zelda_pupil[idx] / clear_pupil[idx]
            zelda_norm = zelda_norm.squeeze()
            zelda_norm[~pup] = 0

            # mask_diffraction_prop array contains the mask diffracted properties:
            #  - [0] reference wave
            #  - [1] dephasing term
            if nwave == 1:
                cwave = wave[0]
                reference_wave = mask_diffraction_prop[0][0]
                expi = mask_diffraction_prop[0][1]
            else:
                cwave = wave[idx]
                reference_wave = mask_diffraction_prop[idx][0]
                expi = mask_diffraction_prop[idx][1]

            # determinant calculation
            delta = (expi.imag)**2 - 2*(reference_wave-1) * (1-expi.real)**2 - \
                    ((1-zelda_norm) / reference_wave) * (1-expi.real)
            delta = delta.real
            delta[~pup] = 0

            # check for negative values
            neg_values = ((delta < 0) & pup)
            neg_count  = neg_values.sum()
            ratio = neg_count / pup.sum() * 100

            if (silent is False):
                print('Negative values: {0} ({1:0.3f}%)'.format(neg_count, ratio))

            # too many nagative values
            if (ratio > 1):
                raise NameError('Too many negative values in determinant (>1%)')

            # replace negative values by 0
            delta[neg_values] = 0

            # phase calculation
            theta = (1 / (1-expi.real)) * (-expi.imag + np.sqrt(delta))
            theta[~pup] = 0

            # optical path difference in nm
            kw = 2*np.pi / cwave
            opd_nm = (1/kw) * theta * 1e9

            # statistics
            if (silent is False):
                print('OPD statistics:')
                print(' * min = {0:0.2f} nm'.format(opd_nm.min()))
                print(' * max = {0:0.2f} nm'.format(opd_nm.max()))
                print(' * std = {0:0.2f} nm'.format(opd_nm.std()))        

            # save
            zelda_pupil[idx] = opd_nm

        # variable name change
        opd_nm = zelda_pupil

        return opd_nm
