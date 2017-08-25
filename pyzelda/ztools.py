# -*- coding: utf-8 -*-
'''
pyZELDA utility methods

arthur.vigan@lam.fr
mamadou.ndiaye@oca.eu
'''

import os
import numpy as np

import pyzelda.utils.mft as mft
import pyzelda.utils.imutils as imutils
import pyzelda.utils.aperture as aperture
import pyzelda.utils.circle_fit as circle_fit

import poppy.zernike as zernike
import scipy.ndimage as ndimage

from astropy.io import fits


def number_of_frames(path, data_files):
    '''
    Returns the total number of frames in a sequence of files

    Parameters
    ----------
    path : str
        Path to the directory that contains the FITS files
    
    data_files : str
        List of files that contains the data, without the .fits

    Returns
    -------
    nframes_total : int
        Total number of frames
    '''
    if type(data_files) is not list:
        data_files = [data_files]
    
    nframes_total = 0
    for fname in data_files:
        img = fits.getdata(os.path.join(path, fname+'.fits'))
        if img.ndim == 2:
            nframes_total += 1
        elif img.ndim == 3:
            nframes_total += img.shape[0]
            
    return nframes_total  


def load_data(path, data_files, width, height, origin):
    '''
    read data from a file and check the nature of data (single frame or cube) 

    Parameters:
    ----------
    path : str
        Path to the directory that contains the FITS files
    
    data_files : str
        List of files that contains the data, without the .fits

    width : int
        Width of the detector window to be extracted
    
    height : int
        Height of the detector window to be extracted
    
    origin : tuple
        Origin point of the detector window to be extracted in the raw files
    
    Returns
    -------
    clear_cube : array_like
        Array containing the collapsed data    
    '''

    # make sure we have a list
    if type(data_files) is not list:
        data_files = [data_files]

    # get number of frames
    nframes_total = number_of_frames(path, data_files)

    # load data
    data_cube = np.zeros((nframes_total, height, width))
    frame_idx = 0
    for fname in data_files:
        data = fits.getdata(path+fname+'.fits')        
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        
        nframes = data.shape[0]
        data_cube[frame_idx:frame_idx+nframes] = data[:, origin[1]:origin[1]+height, origin[0]:origin[0]+width]
        frame_idx += nframes
                
    return data_cube


def pupil_center(clear_pupil, center_method):
    '''
    find the center of the clear pupil
  
    Parameters:
    ----------

    clear_pupil : array_like
        Array containing the collapsed clear pupil data

    center_method : str, optional
        Method to be used for finding the center of the pupil:
         - 'fit': least squares circle fit (default)
         - 'com': center of mass

    Returns
    -------	
    c : vector_like
        Vector containing the (x,y) coordinates of the center in 1024x1024 raw data format	
    '''

    # recenter
    tmp = clear_pupil / np.max(clear_pupil)
    tmp = (tmp >= 0.2).astype(int)

    if (center_method == 'fit'):
        # circle fit
        kernel = np.ones((10, 10), dtype=int)
        tmp = ndimage.binary_fill_holes(tmp, structure=kernel)

        kernel = np.ones((3, 3), dtype=int)
        tmp_flt = ndimage.binary_erosion(tmp, structure=kernel)

        diff = tmp-tmp_flt
        cc = np.where(diff != 0)

        cx, cy, R, residuals = circle_fit.least_square_circle(cc[0], cc[1])
        c = np.array((cx, cy))
        c = np.roll(c, 1)
    elif (center_method == 'com'):
        # center of mass (often fails)
        c = np.array(ndimage.center_of_mass(tmp))
        c = np.roll(c, 1)
    else:
        raise NameError('Unkown centring method '+center_method)

    print('Center: {0:.2f}, {1:.2f}'.format(c[0], c[1]))
        
    return c


def recentred_data_cubes(path, data_files, dark, dim, center, collapse, origin):
    '''
    Read data cubes from disk and recenter them

    Parameters
    ----------
    path : str
        Path to the directory that contains the TIFF files
    
    data_files : str
        List of files to read, without the .fits
    
    dark : array_like
        Dark frame to be subtracted to all images

    dim : int, optional
        Size of the final arrays

    center : vector_like
        Center of the pupil in the images

    collapse : bool
        Collapse or not the cubes
    
    origin : tuple
        Origin point of the detector window to be extracted in the raw files
    
    '''
    center = np.array(center)
    cint = center.astype(np.int)
    cc   = dim//2
        
    # read zelda pupil data (all frames)
    if type(data_files) is not list:
        data_files = [data_files]

    # determine total number of frames
    nframes_total = number_of_frames(path, data_files)

    ext = 5
    data_cube = np.empty((nframes_total, dim+2*ext, dim+2*ext))
    frame_idx = 0
    for fname in data_files:
        # read data
        data = fits.getdata(path+fname+'.fits')
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        nframes = data.shape[0]
        data_cube[frame_idx:frame_idx+nframes] = data[:,
                                                      origin[1]+cint[1]-cc-ext:origin[1]+cint[1]+cc+ext,
                                                      origin[0]+cint[0]-cc-ext:origin[0]+cint[0]+cc+ext]
        frame_idx += nframes
        
        del data

    # collapse if needed
    if collapse:
        data_cube = data_cube.mean(axis=0, keepdims=True)

    # clean and recenter images
    dark_sub = dark[cint[1]-cc-ext:cint[1]+cc+ext, cint[0]-cc-ext:cint[0]+cc+ext]
    for idx, img in enumerate(data_cube):
        img = img - dark_sub

        img = imutils.sigma_filter(img, box=5, nsigma=3, iterate=True)
        img = imutils.shift(img, cint-center-ext)
        
        data_cube[idx] = img
    
    data_cube = data_cube[:, :dim, :dim]
        
    return data_cube


def refractive_index(wave, substrate):
    '''
    Compute the refractive index of a subtrate at a given wavelength, 
    using values from the refractice index database:
    https://refractiveindex.info/
    
    Parameters
    ----------    
    wave: float 
        wavelength in m
    
    substrate: string 
        Name of the substrate
    
    Returns
    -------
    
    n: the refractive index value using the Sellmeier formula

    '''
    # convert wave from m to um
    wave = wave*1e6 
    
    if substrate == 'fused_silica':
        params = {'B1': 0.6961663, 'B2': 0.4079426, 'B3': 0.8974794, 
                  'C1': 0.0684043, 'C2': 0.1162414, 'C3': 9.896161,
                  'wavemin': 0.21, 'wavemax': 3.71}
    else:
        raise ValueError('Unknown substrate {0}!'.format(substrate))
    
    if (wave > params['wavemin']) and (wave < params['wavemax']):
        n = np.sqrt(1 + params['B1']*wave**2/(wave**2-params['C1']**2) +
                params['B2']*wave**2/(wave**2-params['C2']**2) +
                params['B3']*wave**2/(wave**2-params['C3']**2))
    else:
        raise ValueError('Wavelength is out of range for the refractive index')
        
    return n


def create_reference_wave(mask_diameter, mask_depth, mask_substrate, pupil_diameter, Fratio, wave):
    '''
    Simulate the ZELDA reference wave

    Parameters
    ----------

    mask_diameter : float
        Mask physical diameter, in m.
    
    mask_depth : float
        Mask physical depth, in m.
    
    mask_substrate : str
        Mask substrate

    pupil_diameter : int
        Instrument pupil diameter, in pixel.

    Fratio : float
        F ratio at the mask focal plane    

    
    wave : float, optional
        Wavelength of the data, in m.
    
    Returns
    -------
    reference_wave : array_like
        Reference wave as a complex array

    expi : complex
        Phasor term associated  with the phase shift
    '''

    # ++++++++++++++++++++++++++++++++++
    # Zernike mask parameters
    # ++++++++++++++++++++++++++++++++++

    # physical diameter and depth, in m
    d_m = mask_diameter
    z_m = mask_depth

    # substrate refractive index
    n_substrate = refractive_index(wave, mask_substrate)

    # R_mask: mask radius in lam0/D unit
    R_mask = 0.5*d_m / (wave * Fratio)

    # ++++++++++++++++++++++++++++++++++
    # Dimensions
    # ++++++++++++++++++++++++++++++++++

    # mask sampling in the focal plane
    D_mask_pixels = 300

    # entrance pupil radius
    R_pupil_pixels = pupil_diameter/2

    # ++++++++++++++++++++++++++++++++++
    # Numerical simulation part
    # ++++++++++++++++++++++++++++++++++

    # --------------------------------
    # plane A (Entrance pupil plane)

    # definition of m1 parameter for the Matrix Fourier Transform (MFT)
    # here equal to the mask size
    m1 = 2*R_mask

    # defintion of the electric field in plane A in the absence of aberrations
    ampl_PA_noaberr = aperture.disc(pupil_diameter, R_pupil_pixels, cpix=True, strict=True)

    # --------------------------------
    # plane B (Focal plane)

    # calculation of the electric field in plane B with MFT within the Zernike
    # sensor mask
    ampl_PB_noaberr = mft.mft(ampl_PA_noaberr, pupil_diameter, D_mask_pixels, m1)

    # restriction of the MFT with the mask disk of diameter D_mask_pixels/2
    ampl_PB_noaberr = ampl_PB_noaberr * aperture.disc(D_mask_pixels, D_mask_pixels, diameter=True, cpix=True, strict=True)

    # normalization term using the expression of the field in the absence of aberrations without mask
    norm_ampl_PC_noaberr = 1/np.max(np.abs(ampl_PA_noaberr))

    # --------------------------------
    # plane C (Relayed pupil plane)

    # mask phase shift phi (mask in transmission)
    phi = 2*np.pi*(n_substrate-1)*z_m/wave

    # phasor term associated  with the phase shift
    expi = np.exp(1j*phi)

    # --------------------------------
    # definition of parameters for the phase estimate with Zernike

    # b1 = reference_wave: parameter corresponding to the wave diffracted by the mask in the relayed pupil
    reference_wave = norm_ampl_PC_noaberr * mft.mft(ampl_PB_noaberr, D_mask_pixels, pupil_diameter, m1) * \
                     aperture.disc(pupil_diameter, R_pupil_pixels, cpix=True, strict=True)

    return reference_wave, expi


def zernike_expand(opd, nterms=32):
    '''
    Expand an OPD map into Zernike polynomials

    Parameters
    ----------
    opd : array_like
        OPD map in nanometers
    
    nterms : int, optional
        Number of polynomials used in the expension. Default is 15

    Returns
    -------
    basis : array_like
        Cube containing the array of 2D polynomials
    
    coeffs : vector_like
        Vector with the coefficients corresponding to each polynomial
    
    reconstructed_opd : array_like
        Reconstructed OPD map using the basis and determined coefficients
    '''
    
    print('Zernike decomposition')
    
    if opd.ndim == 2:
        opd = opd[np.newaxis, ...]
    nopd = opd.shape[0]
    
    Rpuppix = opd.shape[-1]/2

    # rho, theta coordinates for the aperture
    rho, theta = aperture.coordinates(opd.shape[-1], Rpuppix, cpix=True, strict=True, outside=np.nan)

    wgood = np.where(np.isfinite(rho))
    ngood = (wgood[0]).size

    wbad = np.where(np.logical_not(np.isfinite(rho)))
    rho[wbad]   = 0
    theta[wbad] = 0

    # create the Zernike polynomiales basis
    basis  = zernike.zernike_basis(nterms=nterms, rho=rho, theta=theta)

    coeffs = np.zeros((nopd, nterms))
    reconstructed_opd = np.zeros_like(opd)
    for i in range(nopd):
        # determines the coefficients
        coeffs_tmp = [(opd[i] * b)[wgood].sum() / ngood for b in basis]
        coeffs[i]  = np.array(coeffs_tmp)
        
        # reconstruct the OPD
        for z in range(nterms):
            reconstructed_opd[i] += coeffs_tmp[z] * basis[z, :, :]

    return basis, coeffs, reconstructed_opd