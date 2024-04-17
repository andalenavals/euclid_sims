import logging
import numpy as np
import pandas as pd
import os
import json
import astropy.table
import pickle
from astropy.io import fits
import datetime
import fitsio
import yaml
import ray
RAY=False
if RAY:
    import ray.util.multiprocessing as multiprocessing
else:
    import multiprocessing

logger = logging.getLogger(__name__)



def open_fits(filename, hdu=1):
    logger.debug("trying to read %s"%(filename))
    assert os.path.isfile(filename)
    try:
        with fits.open(filename) as hdul:
            hdul.verify('fix')
            cat=hdul[hdu].data
            cat = pd.DataFrame(cat.astype(cat.dtype.newbyteorder('=')))
            hdul.close()
    except Exception as e:
        logger.info(str(e))
        logger.info("Unable to read existing measurement catalog %s, trying with fitsio"%(filename))
        try:
            cat=fitsio.read(filename, ext=hdu)
            cat = pd.DataFrame(cat.astype(cat.dtype.newbyteorder('=')))
        except Exception as f:
            logger.info(str(f))
            logger.info("Unable to read existing measurement catalog"%(filename))
            raise
    logger.debug("Done reading")
    return cat
           

def _run(workerfunction, wslist, ncpu):
        """
        Wrapper around multiprocessing.Pool with some verbosity.
        """
        if RAY: ray.init(_temp_dir="/vol/euclidraid5/data/aanavarroa/")
        
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
        
        logger.info("Starting %i jobs using %i CPUs" % (len(wslist), ncpu))
        
        if ncpu == 1: # The single process way (MUCH MUCH EASIER TO DEBUG...)
                list(map(workerfunction, wslist))
        
        else:
                '''
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(workerfunction, wslist)
                pool.close()
                pool.join()
                '''
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(workerfunction, wslist)
                pool.close()
                pool.join()
                
        endtime = datetime.datetime.now()
        logger.info("Done, the total running time was %s" % (str(endtime - starttime)))


def parallel_map(process_task, wslist, ncpu):
    # Create a pool of tasks
    total_tasks = len(wslist)
    tasks = [i for i in range(total_tasks)]

    if total_tasks< ncpu:
        ncpu=total_tasks
        
    logger.info("Doing %i jobs using %i cpus"%(total_tasks,ncpu))
    # Create a list to keep track of which CPU is processing which task
    cpu_tasks = [None] * ncpu

    while tasks:
        # Check for available CPUs
        available_cpus = [i for i, task in enumerate(cpu_tasks) if task is None]

        if available_cpus:
            # Assign tasks to available CPUs
            for cpu_index in available_cpus:
                if tasks:
                    task = tasks.pop(0)
                    cpu_tasks[cpu_index] = process_task.remote(wslist[task])

        # Wait for any task to finish
        if len(tasks)==0: num_returns=ncpu
        else: num_returns=1
        finished_task, _ = ray.wait(cpu_tasks, num_returns=num_returns)
        finished_task = finished_task[0]

        # Retrieve result and print
        #result = ray.get(finished_task)
        #print(result)

        # Free up CPU for next task
        cpu_tasks[cpu_tasks.index(finished_task)] = None
    logger.info("Parallel map of jobs finished!!")
        
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
def makepicklecat(filename, typecat="tp", picklefilename=None, matchpairs=False):
    from .variables import GALCOLS,STARCOLS,SEXFEATS,PSFCOLS,MEASCOLS,MATCHCOLS,NEISCOLS,COLS1D,COLS2D,neicols_tru,neicols_sex,neicols_ada, COLS1D_PICKLE, COLS2D_PICKLE
    #get3Ddata_constimg was taking to much time it is more efficient to save the astropy table pickle
    '''
    typecat: type of catalog. It can be 'tp' or 'tw'
    '''
    logger.info("Making pickle catalog")
    
    
    PREDS=["pre_s1", "pre_s2","pre_s1w", "pre_s2w","pre_s1_var", "pre_s2_var" ]+['s1_pred', 's2_pred', 'w1_pred', 'w2_pred', 'm1_pred', 'm2_pred']
  
    if (typecat=="tp"):
        cols1d = COLS1D +[ "cat_id"]+PSFCOLS + ["ws_true", "wi_true"] + GALCOLS
        cols2d = COLS2D +["flag","img_id"]+["subcase_id", "ICS", "ICS2"] + SEXFEATS + PREDS + NEISCOLS+ MEASCOLS
    if (typecat=="tw"):
        cols1d=COLS1D_PICKLE
        cols2d=COLS2D_PICKLE
        # tru_gal_density is 2d only because the selection is done by realization so we need 2d for plots (or training)
        
    picklecat(filename, cols1d=cols1d, cols2d=cols2d,picklefilename=picklefilename, matchpairs=matchpairs, typecat=typecat)
   
def picklecat(filename, cols1d=None, cols2d=None,picklefilename=None, matchpairs=False,typecat=None):
    #get3Ddata_constimg was taking to much time it is more efficient to save the astropy table pickle
    '''
    typecat: type of catalog. It can be 'tp' or 'tw'
    '''
    cat_df=open_fits(filename)
    # only using existing cols in the fict catalog
    cols1d=[ col for col in cols1d if col in cat_df.columns]
    cols2d=[ col for col in cols2d if col in cat_df.columns]
    
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
        vocat=match_pairs(vocat, cols2d=cols2d)
        filename=picklefilename.replace(os.path.splitext(picklefilename)[1], "_matchpairs.pkl")
        with open(filename, 'wb') as handle:
            pickle.dump(vocat, handle, -1)


def match_pairs(vocat, cols2d=None):
    logger.info("Matching pairs in tp-like catalog")
    assert "tru_g1" in vocat.colnames
    assert "tru_g2" in vocat.colnames
    if cols2d is None: cols2d=[ c for c in vocat.colnames if vocat[c].ndim==2]
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
        logger.debug(i,np.sum(case["tru_g1"]), len(indxtomask))
    return vocat
            
def get_cols1d_cols2d_extracols(typegroup, cattype, meastype='adamom'):
    from .variables import GALCOLS,STARCOLS,SEXFEATS,PSFCOLS,MEASCOLS,MATCHCOLS,NEISCOLS,COLS1D,COLS2D,neicols_tru,neicols_sex,neicols_ada  
    if (typegroup =='tp'):
        cols1d = COLS1D+GALCOLS + PSFCOLS 
        if cattype=='tru':
            extracols=COLS2D+ neicols_tru
        if cattype=='sex':
            extracols= SEXFEATS + neicols_sex+COLS2D
    if (typegroup =='tw'):
        cols1d=COLS1D + PSFCOLS
        tru_cols2d=[ "obj_id", "ics"]+COLS2D+ GALCOLS+STARCOLS
        if cattype=='tru':
            extracols=tru_cols2d+ neicols_tru
        if cattype=='sex':
            extracols=tru_cols2d+SEXFEATS+neicols_sex +["ICS2"]
            
    cols2d = MEASCOLS+NEISCOLS+neicols_ada+ extracols+MATCHCOLS
    return cols1d, cols2d, extracols
 

def makedir(outpath):
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

def readyaml(filename):
    try:
        with open(filename) as file:
            aux= yaml.load(file, Loader=yaml.FullLoader)
            return aux
    except OSError :
            with open(filename) as file: raise

def write_to_file(line, filename):
    """Function to write a line to a file."""
    with open(filename, 'a') as f:
        f.write(line + '\n')
def read_integers_from_file(filename):
    """Function to read integers from a file and return a list of them."""
    integers = []
    if filename is None: return integers
    if not os.path.isfile(filename): return integers
    with open(filename, 'r') as f:
        for line in f:
            integers.append(int(line.strip()))
    return integers
        
def group_measurements(filename, inputcat, workdir, picklecat=True):
    logger.info("Grouping catalogs")
    cat=open_fits(inputcat,hdu=1)

    alldata=[]
    for i, row in cat.iterrows():
         meascatfile=os.path.join(workdir,"cat%i.fits"%(i))
         if os.path.isfile(meascatfile):
             mcat=open_fits(meascatfile) #2d data
             for col in set(cat.columns)-set(["psf_file"]) :
                mcat[col] = row[col]
         else:
             continue
         alldata.append(mcat)

    alldata_df=pd.concat(alldata, ignore_index=True)
    alldata_df=alldata_df.sort_values(by=['cat_id'], ignore_index=True)
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)
    if picklecat: makepicklecat(filename, picklefilename=filename.replace(".fits",".pkl"))
    logger.info("Grouping catalogs finished!!")

