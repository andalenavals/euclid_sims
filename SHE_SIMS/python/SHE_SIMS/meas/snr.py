"""
This measfct adds a SNR to the catalog, computed from existing columns.
It does not measure the images.

"""

import fitsio
import pandas as pd
import numpy as np
import copy

import logging
logger = logging.getLogger(__name__)

def measfct(catalog, prefix="", fluxcol="adamom_flux", sigmacol="adamom_sigma", stdcol="skymad", sizecol=None, gain=None, gaincol=None, aper=3.0, subsample_nbins=1):
        """
        We assume here that the images are in ADU, and that 
        gain is given in electron / ADU.
        
        :param fluxcol:
        :param sigmacol:
        :param stdcol:
        :param gain:
        :param gaincol:
        
        :param aper: the radius of the effective area is hlr * aper. So aper = 3, the default value, means an aperture with a diameter of 3 times the half-light-diameter.
                This gives values close to FLUX_AUTO / FLUXERR_AUTO
        
        
        """
        bad_flags=[-999,-888]
        
        if gain != None and gaincol != None:
                raise RuntimeError("Please provide either gain or gaincol, not both!")
                
        output = fitsio.read(catalog)
        output= output.astype(output.dtype.newbyteorder('='))
        output = pd.DataFrame(output)
        
        if gain != None:
                gain = np.abs(gain)
                logger.info("Now computing SNR assuming a gain of {0:.2} electrons per ADU.".format(gain))        
                sourcefluxes = np.where(sum([output[fluxcol]==f for f in bad_flags]), -999, output[fluxcol] * gain)
                skynoisefluxes = np.where(sum([output[fluxcol]==f for f in bad_flags]), -999, (output[stdcol] * gain) ** 2)# per pixel
        
        elif gaincol != None:
                logger.info("Now computing SNR using the gain from column '{}'".format(gaincol))        
                gain = np.abs(output[gaincol])
                sourcefluxes = np.where(sum([output[fluxcol]==f for f in bad_flags]), -999, output[fluxcol] * gain)
                skynoisefluxes = np.where(sum([output[fluxcol]==f for f in bad_flags]), -999, (output[stdcol] * gain) ** 2)# per pixel
        
        else:
                raise RuntimeError("Please provide a gain or gaincol!")
                
        
        if sizecol is None:
                areas = np.where(sum([output[sigmacol]==f for f in bad_flags]), -999, np.pi * (output[sigmacol] * aper * 1.1774*(1./subsample_nbins)) ** 2) # 1.1774 x sigma = r half light
        else: 
                areas = np.where(sum([output[sizecol]==f for f in bad_flags]), -999,output[sizecol])
        
        noises = np.sqrt(sourcefluxes + areas * skynoisefluxes)
        
        snrcol = prefix + "snr"

        #output[snrcol] = np.where(sourcefluxes==-999, -999, sourcefluxes / noises)
        output[snrcol] = np.where(sourcefluxes==-999, -999, np.divide(sourcefluxes, noises, out=np.zeros_like(sourcefluxes), where=noises!=0) )

        fitsio.write(catalog, output.to_records(index=False), clobber=True)

def measfct2(catalog, prefix="", fluxcol="adamom_flux", sigmacol="adamom_sigma", stdcol="skymad", sizecol=None, gain=None, gaincol=None, aper=3.0):
        """
        We assume here that the images are in ADU, and that 
        gain is given in electron / ADU.
        
        :param fluxcol:
        :param sigmacol:
        :param stdcol:
        :param gain:
        :param gaincol:
        
        :param aper: the radius of the effective area is hlr * aper. So aper = 3, the default value, means an aperture with a diameter of 3 times the half-light-diameter.
                This gives values close to FLUX_AUTO / FLUXERR_AUTO
        
        
        """
        bad_flags=[-999,-888]
        
        if gain != None and gaincol != None:
                raise RuntimeError("Please provide either gain or gaincol, not both!")
                
        output = fitsio.read(catalog)
        output= output.astype(output.dtype.newbyteorder('='))
        output = pd.DataFrame(output)
        
        if gain != None:
                gain = np.abs(gain)
                logger.info("Now computing SNR assuming a gain of {0:.2} electrons per ADU.".format(gain))        
                sourcefluxes = np.where(output[fluxcol].isin(bad_flags), -999, output[fluxcol] * gain)
                skynoisefluxes = np.where(output[fluxcol].isin(bad_flags), -999, (output[stdcol] * gain) ** 2)# per pixel
        
        elif gaincol != None:
                logger.info("Now computing SNR using the gain from column '{}'".format(gaincol))        
                gain = np.abs(output[gaincol])
                sourcefluxes = np.where(output[fluxcol].isin(bad_flags), -999, output[fluxcol] * gain)
                skynoisefluxes = np.where(output[fluxcol].isin(bad_flags), -999, (output[stdcol] * gain) ** 2)# per pixel
        
        else:
                raise RuntimeError("Please provide a gain or gaincol!")
                
        
        if sizecol is None:
                areas = np.where(output[sigmacol].isin(bad_flags), -999, np.pi * (output[sigmacol] * aper * 1.1774) ** 2) # 1.1774 x sigma = r half light
        else: 
                areas = np.where(output[sizecol].isin(bad_flags), -999,output[sizecol])
        
        noises = np.sqrt(sourcefluxes + areas * skynoisefluxes)
        
        snrcol = prefix + "snr"

        output[snrcol] = np.where(sourcefluxes==-999, -999, sourcefluxes / noises)

        fitsio.write(catalog, output.to_records(index=False), clobber=True)

        
def measfct3(catalog, prefix="", fluxcol="adamom_flux", sigmacol="adamom_sigma", stdcol="skymad", sizecol=None, gain=None, gaincol=None, aper=3.0):
        """
        We assume here that the images are in ADU, and that 
        gain is given in electron / ADU.
        
        :param fluxcol:
        :param sigmacol:
        :param stdcol:
        :param gain:
        :param gaincol:
        
        :param aper: the radius of the effective area is hlr * aper. So aper = 3, the default value, means an aperture with a diameter of 3 times the half-light-diameter.
                This gives values close to FLUX_AUTO / FLUXERR_AUTO
        
        
        """
        bad_flags=[-999,-888]
        
        if gain != None and gaincol != None:
                raise RuntimeError("Please provide either gain or gaincol, not both!")
                
        output = fitsio.read(catalog)
        output= output.astype(output.dtype.newbyteorder('='))
        output = pd.DataFrame(output)

        calc_cols = [fluxcol, sigmacol, stdcol,  sizecol,  gaincol]
        calc_cols = [i for i in calc_cols if i]

        output.mask(output[calc_cols].isin(bad_flags), inplace=True)
        
        if gain != None:
                gain = np.abs(gain)
                logger.info("Now computing SNR assuming a gain of {0:.2} electrons per ADU.".format(gain))        
                sourcefluxes = output[fluxcol]* gain
                skynoisefluxes = (output[stdcol] * gain) ** 2# per pixel
        
        elif gaincol != None:
                logger.info("Now computing SNR using the gain from column '{}'".format(gaincol))        
                gain = np.abs(output[gaincol])
                sourcefluxes = output[fluxcol] * gain
                skynoisefluxes =  (output[stdcol] * gain) ** 2# per pixel
        
        else:
                raise RuntimeError("Please provide a gain or gaincol!")
                
        
        if sizecol is None:
                areas =  np.pi * (output[sigmacol] * aper * 1.1774) ** 2 # 1.1774 x sigma = r half light
        else: 
                areas = output[sizecol]
        
        noises = np.sqrt(sourcefluxes + areas * skynoisefluxes)
        
        snrcol = prefix + "snr"

        output[snrcol] = sourcefluxes / noises
        output[snrcol].where(output[snrcol].notna(), -999, inplace=True)

        fitsio.write(catalog, output.to_records(index=False), clobber=True)

        
def add_fluxsnr(catalog):
    output = fitsio.read(catalog)
    output= output.astype(output.dtype.newbyteorder('='))
    output = pd.DataFrame(output)
    output["SNR_WIN2"]=output["FLUX_WIN"]/output["FLUXERR_WIN"]      
    fitsio.write(catalog, output.to_records(index=False), clobber=True)              
