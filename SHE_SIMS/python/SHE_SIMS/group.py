import numpy as np
import pandas as pd
import astropy
from .utils import open_fits
from astropy.io import fits
import glob
import os
import fitsio

import logging
logger = logging.getLogger(__name__)

MINOBJS=1

def cantor_pair(k1,k2):
    return int(0.5*(k1+k2)*(k1+k2+1)+k2)

def append_measurements(meascat_dfs, measimg, nimgs,basename, cat_id,ext, label, hdu=1 ):
    for img_id in range(nimgs):
        logger.info("Grouping img %i"%(img_id) )
        if label=="adamom_":
            meascatpath = os.path.join(measimg,"%s_img%i_galimg%s_meascat.fits"%(basename,img_id,ext))
        elif label=="mf_":
            meascatpath = os.path.join(measimg,"%s_img%i_galimg%s_cat.fits"%(basename,img_id,ext))

        try:
            meascat_df=open_fits(meascatpath, hdu=hdu)
        except:
            continue #raise

        if len(meascat_df)==0: 
            logger.info("Image %s is empty"%(meascatpath))
            #assert os.path.isfile(meascatpath)
            #os.remove(meascatpath)
            continue
            
        if "obj_id" in meascat_df.columns:
            c=cantor_pair(cat_id,img_id)
            real_obj_id= np.vectorize(cantor_pair)(c, meascat_df["obj_id"])
            meascat_df["real_obj_id"]=real_obj_id
            
        #this is required only to include neighbor features
        if len(meascat_df)<MINOBJS:
            logger.info("Skipping img %s in grouping number of objects < %i"%(meascatpath,MINOBJS))
            continue
        
        meascat_df["img_id"] = img_id
            
        visarea=(4000*0.1/60)**2 #arcmin^-2
        meascat_df["gal_density"] = len(meascat_df)/visarea
        meascat_dfs.append(meascat_df)

#Only this one used at the moment    
def cats_constimg(simdir, measdir, cols2d=["adamom_flux"], cols1d=["tru_s1", "tru_s2"], label="adamom_", filename=None, base_pair=True, rot_pair=False, stars=False, nsubcases=0, subcasecol=None, cattype="sex"):
    '''
    cols2d: columns that change among realizations
    trucols: columns that are constant among realizations 
    rot_pair: include rot_pair in the measurement 
    nsubcases: number of subcases to be included
    subcascol: measurement col to get percentiles
    '''
    
    imgdirs_meas_basename = sorted( [ "".join(os.path.basename(f).rsplit("_img", 1)) for f in glob.glob(os.path.join(measdir,'*_img') )] )
    simcats_basename = sorted( [os.path.basename(f).replace("_cat.fits", "") for f in glob.glob(os.path.join(simdir,'*_cat.fits') )] )
    if len(imgdirs_meas_basename) != len( set(imgdirs_meas_basename).intersection(simcats_basename)  ):
        raise RuntimeError("Number of sim catalogs %i does not match number of measurement catalogs %i"%( len(imgdirs_meas_basename), len(simcats_basename) ))

    alldata = []
    for cat_id, basename in enumerate(simcats_basename):
        measimg = os.path.join(measdir,'%s_img'%(basename))
        trucatpath = os.path.join(simdir,'%s_cat.fits'%(basename))
        with fits.open(trucatpath) as hdul:
            hdul.verify('fix')
            trucat=hdul[1].data
            trucat_const =hdul[2].data
        #trucat = fitsio.read(trucatpath, ext=1)

        nimgs = max(trucat["img_id"]) + 1
        
        meascat_dfs=[]
        if base_pair:
            append_measurements(meascat_dfs,measimg,nimgs,basename, cat_id,'', label, hdu=1)
            if stars&(cattype=="tru"):
                append_measurements(meascat_dfs,measimg,nimgs,basename, cat_id,'', label, hdu=2)
        if rot_pair:
            append_measurements(meascat_dfs,measimg,nimgs,basename, cat_id, '_rot', label, hdu=1)
            if stars&(cattype=="tru"):
                append_measurements(meascat_dfs,measimg,nimgs,basename, cat_id,'_rot', label, hdu=2)
            

        if len(meascat_dfs)==0:
            logger.info("Skipping case %s in grouping not objects or sextractor catalogs"%(basename))
            continue

        imgsmeascats=pd.concat(meascat_dfs, ignore_index=True)
        if nsubcases==0:
            imgsmeascats["cat_id"] = cat_id
        else:
            assert subcasecol is not None
            splitl=np.array_split(imgsmeascats.sort_values(by=[subcasecol], ignore_index=True).to_records(index=False), nsubcases)
            splitl = [pd.DataFrame(split) for split in splitl]
            auxl=[]
            for i, df in enumerate(splitl):
                df['subcase_id']=i
                df['cat_id']=cat_id*nsubcases+i
                auxl.append(df)
            imgsmeascats=pd.concat(auxl, ignore_index=True)

            assert ~np.any(np.isnan(imgsmeascats['cat_id']))
        

        ## Adding known true features which not require matching (constant in the whole image       
        cols1d=[col for col in cols1d if col in trucat_const.dtype.names]
        for col in cols1d :
                imgsmeascats[col] = trucat_const[col][0]          

        cols2d=[col for col in cols2d if col in meascat_dfs[0].columns]
        if stars&(cattype=="tru"):
            cols2d=list(set(cols2d +[col for col in cols2d if col in meascat_dfs[1].columns]))       
        #Keep only specific columns and ids for future getting 3d and 2d data
        if label == "adamom_":
            listcols=cols2d+cols1d+['flag']+['cat_id', 'img_id']
            if "real_obj_id" in meascat_dfs[0].columns:
                listcols+=["real_obj_id"]
        elif label == "mf_":
            listcols=cols2d+cols1d+['cat_id', 'img_id']
        if nsubcases>0: listcols.append('subcase_id')

        alldata.append(imgsmeascats[listcols]) 

    alldata_df=pd.concat(alldata, ignore_index=True)
    alldata_df=alldata_df.sort_values(by=['cat_id',  'img_id'], ignore_index=True)
    #print(np.sum(alldata_df.loc[ alldata_df["star_flag"]==0, "tru_g1"]),np.sum(alldata_df.loc[alldata_df["star_flag"]==0,"tru_g2"]))
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)
    logger.info("Grouping finished")

#grouping using only realization where there is a good measurement in both galaxies in a pair, measure catalogs must have obj_id
def cats_constimg_rotpair(simdir, measdir, cols2d=["adamom_flux"], cols1d=["tru_s1", "tru_s2"], stars=False,  filename=None, cattype="sex"):
    '''
    idem but but only include realizations in the catalog where both measurements where good 
    #the obj_id is important for this measure therefore only use this function when adamom measures used tru catalog
    
    '''

    label="adamom_"
    imgdirs_meas_basename = sorted( [ "".join(os.path.basename(f).rsplit("_img", 1)) for f in glob.glob(os.path.join(measdir,'*_img') )] )
    simcats_basename = sorted( [os.path.basename(f).replace("_cat.fits", "") for f in glob.glob(os.path.join(simdir,'*_cat.fits') )] )
    if len(imgdirs_meas_basename) != len( set(imgdirs_meas_basename).intersection(simcats_basename)  ):
        raise RuntimeError("Number of sim catalogs %i does not match number of measurement catalogs %i"%( len(imgdirs_meas_basename), len(simcats_basename) ))

    alldata = []
    for cat_id, basename in enumerate(simcats_basename):
        measimg = os.path.join(measdir,'%s_img'%(basename))
        trucatpath = os.path.join(simdir,'%s_cat.fits'%(basename))
        trucat = fitsio.read(trucatpath, ext=1)

        nimgs = max(trucat["img_id"]) + 1
        
        meascat_dfs=[]
        for img_id in range(nimgs):
            logger.info("Grouping img %i"%(img_id) )
          
            meascatpath = os.path.join(measimg,"%s_img%i_galimg_meascat.fits"%(basename,img_id))
            meascatpath_rot = os.path.join(measimg,"%s_img%i_galimg_rot_meascat.fits"%(basename,img_id))
     
            meascat_df=open_fits(meascatpath)
            meascat_rot_df=open_fits(meascatpath_rot)

            #this is required only to include neighbor features
            if len(meascat_df)<MINOBJS:
                logger.info("Skipping img %s in grouping number of objects < %i"%(meascatpath,minobjs))
                continue

            meascat_df["img_id"] = 2*img_id
            meascat_rot_df["img_id"] = 2*img_id+1
            
            
            visarea=(4000*0.1/60)**2 #arcmin^2
            #meascat_df["gal_density"] = (np.max(meascat_df["obj_number"])+1)/visarea
            meascat_df["gal_density"] = len(meascat_df)/visarea
            meascat_rot_df["gal_density"] = len(meascat_rot_df)/visarea


            # this lines remove detection bias
            if stars:
                obj_ids=list(set(meascat_df.loc[meascat_df["star_flag"]==0,"obj_id"]).intersection(set(meascat_rot_df.loc[meascat_rot_df["star_flag"]==0,"obj_id"])))
            else:
                obj_ids=list(set(meascat_df["obj_id"]).intersection(set(meascat_rot_df["obj_id"])))
            meascat_df=meascat_df[meascat_df['obj_id'].isin(obj_ids)]
            meascat_rot_df=meascat_rot_df[meascat_rot_df['obj_id'].isin(obj_ids)]

            c=cantor_pair(cat_id,img_id)
    
            meascat_df["real_obj_id"]=np.vectorize(cantor_pair)(c, meascat_df["obj_id"])
            meascat_rot_df["real_obj_id"]=np.vectorize(cantor_pair)(c, meascat_rot_df["obj_id"])
                
            meascat_dfs.append(meascat_df)
            meascat_dfs.append(meascat_rot_df)

            if stars&(cattype=="tru"):
                meascatstars_df=open_fits(meascatpath, hdu=2)
                meascatstars_rot_df=open_fits(meascatpath_rot, hdu=2)

                meascatstars_df["img_id"] = 2*img_id
                meascatstars_rot_df["img_id"] = 2*img_id+1
            
                meascatstars_df["gal_density"] = len(meascat_df)/visarea
                meascatstars_rot_df["gal_density"] = len(meascat_rot_df)/visarea
                meascatstars_df["real_obj_id"]=np.vectorize(cantor_pair)(c, meascatstars_df["obj_id"])
                meascatstars_rot_df["real_obj_id"]=np.vectorize(cantor_pair)(c, meascatstars_rot_df["obj_id"])

                meascat_dfs.append(meascatstars_df)
                meascat_dfs.append(meascatstars_rot_df)
            

        if len(meascat_dfs)==0:
            logger.info("Skipping case %s in grouping not objects or sextractor catalogs"%(basename))
            continue
        
        imgsmeascats=pd.concat(meascat_dfs, ignore_index=True)
        imgsmeascats["cat_id"] = cat_id
        

        ## Adding known true features which not require matching (constant in the whole image)
        trucat_const = fitsio.read(trucatpath, ext=2)
        cols1d=[col for col in cols1d if col in trucat_const.dtype.names]
        for col in cols1d :
                imgsmeascats[col] = trucat_const[col][0]

        cols2d=[col for col in cols2d if col in meascat_df.columns]
       
    
        #Keep only specific columns and ids for future getting 3d and 2d data
        if label == "adamom_":
            alldata.append(imgsmeascats[cols2d+cols1d+['flag']+['cat_id', 'img_id','real_obj_id']]) 
        elif label == "mf_":
            alldata.append(imgsmeascats[cols2d+cols1d+['cat_id', 'img_id','real_obj_id']]) 
            
    alldata_df=pd.concat(alldata, ignore_index=True)
    alldata_df=alldata_df.sort_values(by=['cat_id',  'img_id'], ignore_index=True)
    #print(np.sum(alldata_df.loc[ alldata_df["star_flag"]==0, "tru_g1"]),np.sum(alldata_df.loc[alldata_df["star_flag"]==0,"tru_g2"]))
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)
  

def trucats(simdir, cols1d=['tru_s1','tru_s2'], cols2d=['tru_g1','tru_g2'], stars=False,filename=None):
    ext='_cat.fits'
    input_catalogs=sorted(glob.glob(os.path.join(simdir, '*%s'%(ext)) ))

    
    alldata=[]
    for cat_id, catname in enumerate(input_catalogs):
        logger.info("Doing cat %s"%(catname))
        cat=open_fits(catname)
        cat["cat_id"] = cat_id
        
        trucat_const = fitsio.read(catname, ext=2)
        cols1d=[col for col in cols1d if col in trucat_const.dtype.names]
        cols2d=[col for col in cols2d if col in cat.columns]
        for col in cols1d :
                cat[col] = trucat_const[col][0]
        alldata.append(cat[cols2d+cols1d+['cat_id', 'img_id']])
    alldata_df=pd.concat(alldata, ignore_index=True)
    alldata_df=alldata_df.sort_values(by=['cat_id',  'img_id'], ignore_index=True)
    


    if stars:
        alldata_stars=[]
        for cat_id, catname in enumerate(input_catalogs):
            logger.info("Doing cat %s"%(catname))
            cat=open_fits(catname,hdu=3)
            cat["cat_id"] = cat_id
            
            trucat_const = fitsio.read(catname, ext=2)
            cols1d=[col for col in cols1d if col in trucat_const.dtype.names]
            cols2d=[col for col in cols2d if col in cat.columns]
            for col in cols1d :
                cat[col] = trucat_const[col][0]
            alldata_stars.append(cat[cols2d+cols1d+['cat_id', 'img_id']])
        alldata_df_stars=pd.concat(alldata, ignore_index=True)
        alldata_df_stars=alldata_df_stars.sort_values(by=['cat_id',  'img_id'], ignore_index=True)


    if filename is None: filename=os.path.join(simdir, 'truegroupcats.fits')
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)
    fitsio.write(filename,  alldata_df_stars.to_records(index=False), clobber=False)
