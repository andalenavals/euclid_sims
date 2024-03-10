import logging
import numpy as np
import pandas as pd
import os
import json
import astropy.table
import pickle
from astropy.io import fits
import multiprocessing
import datetime


logger = logging.getLogger(__name__)



def open_fits(filename, hdu=1):
    logger.debug("trying to read %s"%(filename))
    assert os.path.isfile(filename)
    try:
        with fits.open(filename) as hdul:
            hdul.verify('fix')
            cat=hdul[hdu].data
            cat = cat.astype(cat.dtype.newbyteorder('='))
            return pd.DataFrame(cat)
    except Exception as e:
        logger.info("Unable to read existing measurement catalog %s, redoing it"%(filename))
        logger.info(str(e))
        if os.path.exists(filename):
            logger.info("removing %s"%(filename))
            #os.remove(meascatpath)
            raise

def _run(workerfunction, wslist, ncpu):
        """
        Wrapper around multiprocessing.Pool with some verbosity.
        """
        
        if len(wslist) == 0: # This test is useful, as pool.map otherwise starts and is a pain to kill.
                logger.info("No images to measure.")
                return

        if ncpu == 0:
                try:
                        ncpu = multiprocessing.cpu_count()
                except:
                        logger.warning("multiprocessing.cpu_count() is not implemented!")
                        ncpu = 1
        
        starttime = datetime.datetime.now()
        
        logger.info("Starting the drawing of %i images using %i CPUs" % (len(wslist), ncpu))
        
        if ncpu == 1: # The single process way (MUCH MUCH EASIER TO DEBUG...)
                list(map(workerfunction, wslist))
        
        else:
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(workerfunction, wslist)
                pool.close()
                pool.join()
        
        endtime = datetime.datetime.now()
        logger.info("Done, the total running time was %s" % (str(endtime - starttime)))


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
 
# The functions below are just for memory optimization you do not want to keep all the         
def makepicklecat(filename, typecat="tp", picklefilename=None, matchpairs=True,):
    from .variables import GALCOLS,SEXFEATS,PSFCOLS,MEASCOLS,MATCHCOLS,NEISCOLS,COLS1D,COLS2D,neicols_tru,neicols_sex,neicols_ada
    #get3Ddata_constimg was taking to much time it is more efficient to save the astropy table pickle
    '''
    typecat: type of catalog. It can be 'tp' or 'tw'
    '''
    logger.info("Making pickle catalog")
    
    
    PREDS=["pre_s1", "pre_s2","pre_s1w", "pre_s2w","pre_s1_var", "pre_s2_var" ]+['s1_pred', 's2_pred', 'w1_pred', 'w2_pred', 'm1_pred', 'm2_pred']
  
    if (typecat=="tp"):
        cols1d = COLS1D +[ "cat_id"]+PSFCOLS + ["ws_true", "wi_true"] + GALCOLS
        cols2d = COLS2D +["flag","img_id"]+["subcase_id", "ICS", "ICS2"] + SEXFEATS + PREDS + NEIS+ MEASCOLS
    if (typecat=="tw"):
        cols1d=COLS1D + ["cat_id"] +PSFCOLS
        cols2d=COLS2D+["real_obj_id", "flag", "obj_id", "img_id"]+GALCOLS+["subcase_id", "ICS", "ICS2"]+MATCHCOLS + SEXFEATS + PREDS + NEISCOLS + MEASCOLS +neicols_sex+neicols_ada+neicols_tru
        # tru_gal_density is 2d only because the selection is done by realization so we need 2d for plots (or training)
        
    picklecat(filename, cols1d=cols1d, cols2d=cols2d,picklefilename=picklefilename, matchpairs=matchpairs, typecat=typecat)
   
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

def get_cols1d_cols2d_extracols(typegroup, cattype, meastype='adamom'):
    from .variables import GALCOLS,SEXFEATS,PSFCOLS,MEASCOLS,MATCHCOLS,NEISCOLS,COLS1D,COLS2D,neicols_tru,neicols_sex,neicols_ada    
    if (typegroup =='tp'):
        cols1d = COLS1D+GALCOLS + PSFCOLS 
        if cattype=='tru':
            extracols=COLS2D+ neicols_tru
        if cattype=='sex':
            extracols= SEXFEATS + neicols_sex+COLS2D
    if (typegroup =='tw'):
        cols1d=COLS1D + PSFCOLS
        tru_cols2d=[ "obj_id", "ics"]+COLS2D+ GALCOLS
        if cattype=='tru':
            extracols=tru_cols2d+ neicols_tru
        if cattype=='sex':
            extracols=tru_cols2d+SEXFEATS+neicols_sex +["ICS2"]
            
    cols2d = MEASCOLS+NEISCOLS+neicols_ada+ extracols+MATCHCOLS
    return cols1d, cols2d, extracols
 


