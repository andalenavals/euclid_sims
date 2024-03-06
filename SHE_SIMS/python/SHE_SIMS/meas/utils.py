"""
General helper functions for shape measurements
"""

import fitsio
import pandas as pd
import numpy as np
import galsim
import astropy

import logging
logger = logging.getLogger(__name__)


def mad(nparray):
        """
        The Median Absolute Deviation
        http://en.wikipedia.org/wiki/Median_absolute_deviation

        Multiply this by 1.4826 to convert into an estimate of the Gaussian std.
        """

        return np.median(np.fabs(nparray - np.median(nparray)))



def skystats(stamp, edgewidth=1):
        """
        I measure some statistics of the pixels along the edge of an image or stamp.
        Useful to measure the sky noise, but also to check for problems. Use "mad"
        directly as a robust estimate the sky std.

        :param stamp: a galsim image, usually a stamp

        :returns: a dict containing "std", "mad", "mean" and "med"
        
        Note that "mad" is already rescaled by 1.4826 to be comparable with std.
        """
      
        if isinstance(stamp, galsim.Image):
                a = stamp.array
                # Normally there should be a .transpose() here, to get the orientation right.
                # But in the present case it doesn't change anything, and we can skip it.
        else:
                a = stamp # Then we assume that it's simply a numpy array.
                
        edgepixels = np.concatenate([
                a[:edgewidth, edgewidth:].flatten(), # left
                a[-edgewidth:,edgewidth:].flatten(), # right
                a[:,:edgewidth].flatten(), # bottom
                a[edgewidth:-edgewidth,-edgewidth:].flatten() # top
        ])

        assert len(edgepixels) == 2*edgewidth*(a.shape[0]+a.shape[1]-2*edgewidth)
                
        # And we convert the mad into an estimate of the Gaussian std:
        return {
                "std":np.std(edgepixels), "mad": 1.4826 * mad(edgepixels),
                "mean":np.mean(edgepixels), "med":np.median(edgepixels),
                "stampsum":np.sum(a)
        }


def add_mfshear(catalog):
        """
        Adding mf_shear to gruopmeascat based on FlexiModelFitting of sextractor
        catalog: fitsfile containing catalog
        """
        bad_flags=[-999,-888]

        output = fitsio.read(catalog)
        output = output.astype(output.dtype.newbyteorder('='))
        output = pd.DataFrame(output)

        mf_g = (1 - output['mf_ratio']) / (1 + output['mf_ratio'])
        mf_g1 = mf_g * np.cos(2 * output['mf_angle'])
        mf_g2 = mf_g * np.sin(2 * output['mf_angle'])
        output['mf_g1'] = np.where(output['mf_ratio'].isin(bad_flags)|output['mf_angle'].isin(bad_flags),-999,mf_g1)
        output['mf_g2'] = np.where(output['mf_ratio'].isin(bad_flags)|output['mf_angle'].isin(bad_flags),-999,mf_g2)
        fitsio.write(catalog, output.to_records(index=False), clobber=True)

