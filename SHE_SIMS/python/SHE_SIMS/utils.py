#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Founddation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: python/SHE_KerasML/utils.py

Created on: 10/12/19
Author: Malte Tewes
"""

import logging
import numpy as np
import pandas as pd
import os
import json
import astropy.table
import pickle
from astropy.io import fits

logger = logging.getLogger(__name__)

def shapeit(cat, cols):
    """
    function to add masked values with the porpouse of keeping same dimensions
    """
    #ncases = max(cat["cat_id"]) + 1
    #nreas = max( [len(cat[cat["cat_id"]==case_id]) for case_id in range(ncases)] )
    cases= cat["cat_id"].drop_duplicates()
    ncases=len(cases)
    nreas = max( [len(cat[cat["cat_id"]==case_id]) for case_id in cases] )
    print("SHAPE:", (ncases,nreas,len(cols)  ) )
    outcat = np.full((ncases,nreas,len(cols)  ) ,-999., dtype=float)
    for i,case_id in enumerate(cases):
        auxcat = cat[cat["cat_id"]==case_id]
        outcat[i, :len(auxcat) , : ] = auxcat[cols].to_numpy()
    outcat = np.ma.array(outcat, mask=np.isclose(outcat,-999))
    return outcat
def get3Ddata_constimg(cat, features_cols, targets_cols=None):
    logger.info("Making 3D data")
    features = shapeit(cat,features_cols)
    if targets_cols is not None:
        targets = shapeit(cat,targets_cols)
        targets= targets[:,:1,:]
        return features, targets
    return features


def write_fit(data, file_name, clobber=True):
    import fitsio
    from fitsio import FITS,FITSHDR
 
    if not os.path.isfile(file_name) :
        #clobber false to not delete existing file
        fitsio.write(file_name,  data, clobber=clobber)
    else:
        # appending data to a new header
        fits = FITS(file_name,'rw')
        fits[-1].append(data)
    
        
def makepicklecat(filename, typecat="tp", picklefilename=None):
    #get3Ddata_constimg was taking to much time it is more efficient to save the astropy table pickle
    '''
    typecat: type of catalog. It can be 'tp' or 'tw'
    '''
    logger.info("Making pickle catalog")

    SEXFEATS=["ELONGATION","AWIN_IMAGE", "BWIN_IMAGE" , "THETAWIN_IMAGE", "ERRAWIN_IMAGE", "ERRBWIN_IMAGE", "ERRTHETAWIN_IMAGE" , "CXXWIN_IMAGE", "CYYWIN_IMAGE", "CXYWIN_IMAGE", "ERRCXXWIN_IMAGE", "ERRCYYWIN_IMAGE", "ERRCXYWIN_IMAGE" ]+["ELONGATION_WIN", "ELLIP_WIN", "ELLIP1_WIN","ELLIP2_WIN","ELLIP_AREA","FLUX_WIN","FLUXERR_WIN"]+["X_IMAGE", "Y_IMAGE", "MAG_AUTO", "MAGERR_AUTO","MAG_WIN","SNR_WIN", "FLAGS","FWHM_IMAGE", "MAG_PSF", "PETRO_RADIUS", "KRON_RADIUS",  "SPREADERR_MODEL", "CLASS_STAR", "FLUX_RADIUS"]

    PREDS=["pre_s1", "pre_s2","pre_s1w", "pre_s2w","pre_s1_var", "pre_s2_var" ]+['s1_pred', 's2_pred', 'w1_pred', 'w2_pred', 'm1_pred', 'm2_pred']
    NEIS=np.concatenate([["fracpix%i"%(i),"fracpix_md%i"%(i)] for i in [10,15,20,25,30]]).tolist() +["mpix",  "fracpix", "fracpix_md","r_nmpix", 'galseg_area','obj_number', "gal_density"]
    MEAS=["adamom_%s"%(f) for f in ["flux","g1", "g2", "sigma", "rho4", "x", "y"]]+["skymad","snr"]+['psf_g1_ksb', 'psf_sigma_ksb', 'psf_g2_ksb']+["corr_s1","corr_s2"]+["ngmix_dx", "ngmix_dy","ngmix_e1", "ngmix_e2", "ngmix_T", "ngmix_flux" ]+np.concatenate([[ "ngmix_%s_%s%s"%("moments",f,"corr"),"ngmix_%s_%s%s"%("moments",f,""), "ngmix_%s_%s%s"%("fit",f,"corr")] for f in ["flags","T","T_err","s2n","g1","g2","g1_err","g2_err"]]).tolist()+np.concatenate([[ "ngmix_%s_%s%s"%("moments",f,"corr"),"ngmix_%s_%s%s"%("moments",f,"")] for f in ["flux","flux_err"]]).tolist()
    
    if (typecat=="tp"):
        cols1d = ["tru_s1", "tru_s2", "tru_g", "tru_rad", "tru_sb", "tru_sersicn", "tru_flux", "tru_gal_density", "tru_mag", "tru_sky_level"] +[ "cat_id","fov_pos"]+["psf_adamom_%s"%(f) for f in ["flux","g1", "g2", "sigma", "rho4"] ]+["e1", "e2", "R2", "ccd", "quadrant", "z", "sed"]+[ "%s%s"%("psf_mom_", n) for n in ["flux", "g1", "g2", "sigma", "rho4", "M4_1","M4_2"]]+ ["tru_bulge_flux","tru_disk_flux", "tru_disk_rad","tru_bulge_rad", "tru_disk_inclination", "tru_bulge_sersicn", "tru_disk_scaleheight", "dominant_shape"] + ["ws_true", "wi_true"]
        cols2d = ["x","y", "tru_g1", "tru_g2", "tru_theta"] +["flag","img_id"]+["r_n1", "adamom_flux_n1","adamom_sigma_n1"]+["r_n1_D_fracpix"]+["subcase_id", "ICS", "ICS2"] + SEXFEATS + PREDS + NEIS+ MEAS
    if (typecat=="tw"):
        cols1d=["tru_s1", "tru_s2", "tru_gal_density", "tru_sky_level"] + ["cat_id","fov_pos"]+["psf_adamom_%s"%(f) for f in ["flux","g1", "g2", "sigma", "rho4"] ]+["e1", "e2", "R2", "ccd", "quadrant", "z", "sed"]+[ "%s%s"%("psf_mom_", n) for n in ["flux", "g1", "g2", "sigma", "rho4", "M4_1","M4_2"]]
        cols2d=["tru_g", "tru_rad", "tru_sb", "tru_mag" ,"tru_sersicn", "tru_flux", "x","y", "tru_g1", "tru_g2", "tru_theta", "real_obj_id", "flag", "obj_id", "img_id"]+["tru_bulge_g","tru_theta", "tru_bulge_g1", "tru_bulge_g2", "tru_bulge_flux", "tru_bulge_sersicn", "tru_bulge_rad", "tru_disk_flux", "tru_disk_rad", "tru_disk_inclination", "tru_disk_scaleheight", "dominant_shape"]+["subcase_id", "ICS", "ICS2"]+["tru_r_n1", "tru_r_n2"]+["tru_flux_n1","tru_rad_n1", "tru_mag_n1"]+["tru_flux_n2","tru_rad_n2", "tru_mag_n2"]+["SEX_R_n1", "SEX_R_n2", "SNR_WIN_n1", "SNR_WIN_n2", "MAG_AUTO_n1", "MAG_AUTO_n2"]+["adamom_r_n1", "adamom_flux_n1", "adamom_sigma_n1"]+["adamom_r_n2", "adamom_flux_n2", "adamom_sigma_n2"]+["tru_rad_match", "tru_mag_match", "tru_r_n1_match", "r_match"] + SEXFEATS + PREDS + NEIS + MEAS
        # tru_gal_density is 2d only because the selection is done by realization so we need 2d for plots (or training) 
        
    picklecat(filename, cols1d=cols1d, cols2d=cols2d,picklefilename=picklefilename,typecat=typecat)
   
def picklecat(filename, cols1d=None, cols2d=None,picklefilename=None, matchpairs=True,typecat=None):
    #get3Ddata_constimg was taking to much time it is more efficient to save the astropy table pickle
    '''
    typecat: type of catalog. It can be 'tp' or 'tw'
    '''
    with fits.open(filename) as hdul:
        hdul.verify('fix')
        cat=hdul[1].data
    #cat=fitsio.read(filename)
    cat = cat.astype(cat.dtype.newbyteorder('='))
    cat_df=pd.DataFrame(cat)
    # only using existing cols in the fict catalog
    cols1d=[ col for col in cols1d if col in cat_df]
    cols2d=[ col for col in cols2d if col in cat_df]
    
    m_features = get3Ddata_constimg(cat_df,cols2d )
    t_features = get3Ddata_constimg(cat_df,cols1d)
    length=m_features.shape[0] #this should be ncases
    vocat= astropy.table.Table()
    
    for i, col in enumerate(cols2d):
        c= astropy.table.MaskedColumn(data=m_features[:,:,i].data, mask=m_features[:,:,i].mask ,name=col, dtype=float, length=length)
        vocat.add_column(c)
 
    for i,col in enumerate(cols1d):
        c= astropy.table.MaskedColumn(data=t_features[:,0,i].data, mask=t_features[:,0,i].mask ,name=col, dtype=float, length=length)
        vocat.add_column(c)
        
    with open(picklefilename, 'wb') as handle:
            pickle.dump(vocat, handle, -1)

    if matchpairs&(typecat=="tp")&("tru_g1" in vocat.colnames):
        logger.info("Matching pairs in tp-like catalog")
        assert "tru_g1" in vocat.colnames
        assert "tru_g2" in vocat.colnames
        for i, case in enumerate(vocat):
            if np.isclose(np.sum(case["tru_g1"]), 0.0): continue
            indxtomask=[]
            for j, [g1,g2] in enumerate(zip(case["tru_g1"], case["tru_g2"])):
                if isinstance(g1,np.ma.core.MaskedConstant): continue
                ind1=np.where(np.isclose(case["tru_g1"],-g1))[0]
                ind2=np.where(np.isclose(case["tru_g2"],-g2))[0]
                inter=set(ind1).intersection(set(ind2))
                if len(inter)!=1:
                    indxtomask.append(j)
                else:
                    #print("mask", i,j, ind1,ind2, inter)
                    continue
            for c in cols2d:
                case[c].mask[indxtomask]=True
            print(i,np.sum(case["tru_g1"]), len(indxtomask))
        filename=picklefilename.replace(os.path.splitext(picklefilename)[1], "_matchpairs.pkl")
        with open(filename, 'wb') as handle:
            pickle.dump(vocat, handle, -1)

