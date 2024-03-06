import numpy as np
import pandas as pd
import astropy
import astropy.io.fits
from astropy.io import fits
import glob
import os
import fitsio

import logging
logger = logging.getLogger(__name__)

MINOBJS=1

def cantor_pair(k1,k2):
    return int(0.5*(k1+k2)*(k1+k2+1)+k2)

#Only this one used at the moment    
def cats_constimg(simdir, measdir, cols2d=["adamom_flux"], cols1d=["tru_s1", "tru_s2"], label="adamom_", filename=None, rot_pair=False, nsubcases=0, subcasecol=None):
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

    minobjs=MINOBJS
    alldata = []
    missing_meas=[]
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
        for img_id in range(nimgs):
            logger.info("Grouping img %i"%(img_id) )
            if label=="adamom_":
                meascatpath = os.path.join(measimg,"%s_img%i_galimg_meascat.fits"%(basename,img_id))
            elif label=="mf_":
                meascatpath = os.path.join(measimg,"%s_img%i_galimg_cat.fits"%(basename,img_id))

            if not os.path.exists(meascatpath):
                missing_meas.append(meascatpath)
                continue
                #raise RuntimeError("Measurement catalog %s does not exist! Probably Something went wrong when running sextractor before"%( meascatpath ))
            try:
                with fits.open(meascatpath) as hdul:
                    hdul.verify('fix')
                    meascat=hdul[1].data
                #meascat = fitsio.read(meascatpath)
            except Exception as e:
                logger.info("Unable to read existing measurement catalog %s, redoing it"%(outcat))
                logger.info(str(e))
                if os.path.exists(meascatpath):
                    logger.info("removing %s"%(meascatpath))
                    #os.remove(meascatpath)
                    break
            meascat = meascat.astype(meascat.dtype.newbyteorder('='))
            meascat_df = pd.DataFrame(meascat)
            if "obj_id" in meascat_df.columns:
                c=cantor_pair(cat_id,img_id)
                real_obj_id= np.vectorize(cantor_pair)(c, meascat_df["obj_id"])
                meascat_df["real_obj_id"]=real_obj_id

            #this is required only to include neighbor features
            if len(meascat_df)<minobjs:
                logger.info("Skipping img %s in grouping number of objects < %i"%(meascatpath,minobjs))
                continue
        
            meascat_df["img_id"] = img_id
            
            if "gal_density" in cols2d:
                visarea=(4000*0.1/60)**2 #arcmin^-2
                #meascat_df["gal_density"] = (np.max(meascat_df["obj_number"])+1)/visarea
                meascat_df["gal_density"] = len(meascat_df)/visarea
            meascat_dfs.append(meascat_df)

        if rot_pair:
            for img_id in range(nimgs):
                logger.info("Grouping img %i"%(img_id) )
                if label=="adamom_":
                    meascatpath = os.path.join(measimg,"%s_img%i_galimg_rot_meascat.fits"%(basename,img_id))
                elif label=="mf_":
                    meascatpath = os.path.join(measimg,"%s_img%i_galimg_rot_cat.fits"%(basename,img_id))

                if not os.path.exists(meascatpath):
                    missing_meas.append(meascatpath)
                    continue
                    #raise RuntimeError("Measurement catalog %s does not exist! "%( meascatpath ))

                try:
                    with fits.open(meascatpath) as hdul:
                        hdul.verify('fix')
                        meascat=hdul[1].data
                    #meascat = fitsio.read(meascatpath)
                except Exception as e:
                    logger.info("Unable to read existing measurement catalog %s, redoing it"%(outcat))
                    logger.info(str(e))
                    if os.path.exists(meascatpath):
                        logger.info("removing %s"%(meascatpath))
                        #os.remove(meascatpath)
                        continue
                meascat = meascat.astype(meascat.dtype.newbyteorder('='))
                meascat_df = pd.DataFrame(meascat)

                #this is required only to include neighbor features
                if len(meascat_df)<minobjs:
                    logger.info("Skipping img %s in grouping number of objects < %i"%(meascatpath,minobjs))
                    continue
        
                if "obj_id" in meascat_df.columns:
                    c=cantor_pair(cat_id,img_id)
                    real_obj_id= np.vectorize(cantor_pair)(c, meascat_df["obj_id"])
                    meascat_df["real_obj_id"]=real_obj_id
                
                meascat_df["img_id"] = img_id+nimgs
                
                if "gal_density" in cols2d:
                    visarea=(4000*0.1/60)**2 #arcmin^-2
                    #meascat_df["gal_density"] = (np.max(meascat_df["obj_number"])+1)/visarea
                    meascat_df["gal_density"] = len(meascat_df)/visarea
                meascat_dfs.append(meascat_df)
            

        if len(meascat_dfs)==0:
            logger.info("Skipping case %s in grouping not objects or sextractor catalogs"%(basename))
            continue

        imgsmeascats=pd.concat(meascat_dfs, ignore_index=True)
        if nsubcases==0:
            imgsmeascats["cat_id"] = cat_id
        else:
            assert subcasecol is not None
            splitl=np.array_split(imgsmeascats.sort_values(by=[subcasecol], ignore_index=True), nsubcases)
            auxl=[]
            for i, df in enumerate(splitl):
                df['subcase_id']=i
                df['cat_id']=cat_id*nsubcases+i
                auxl.append(df)
            imgsmeascats=pd.concat(auxl, ignore_index=True)

            #binsize=np.round(imgsmeascats.shape[0]/nsubcases)
            #for i in range(nsubcases):
            #    imgsmeascats.loc[[i for i in range(,)],"cat_id"] = cat_id*nsubcases+i
            
            '''
            binlims = np.array([np.nanpercentile(imgsmeascats[subcasecol].to_numpy(), q) for q in np.linspace(0.0, 100.0, nsubcases+1)])
            for i in range(nsubcases):
                lowlim, uplim= binlims[i], binlims[i+1]
                mask=(imgsmeascats[subcasecol]>=lowlim)&(imgsmeascats[subcasecol]<uplim)
                imgsmeascats.loc[mask ,"subcase_id"] = i
                imgsmeascats.loc[mask ,"cat_id"] = cat_id*nsubcases+i
                print("lims", lowlim, uplim)
                with pd.option_context('display.max_rows', 100, 'display.max_columns', None):
                    print(imgsmeascats[[subcasecol,"cat_id", "subcase_id"]])
            '''
    
            assert ~np.any(np.isnan(imgsmeascats['cat_id']))
        

        ## Adding known true features which not require matching (constant in the whole image)
        #trucat_const = fitsio.read(trucatpath, ext=2)            
        cols1d=[col for col in cols1d if col in trucat_const.dtype.names]
        for col in cols1d :
                imgsmeascats[col] = trucat_const[col][0]          

        cols2d=[col for col in cols2d if col in meascat_df.columns]
    
        #Keep only specific columns and ids for future getting 3d and 2d data
        if label == "adamom_":
            listcols=cols2d+cols1d+['flag']+['cat_id', 'img_id']
            if "real_obj_id" in meascat_df.columns:
                listcols+=["real_obj_id"]
        elif label == "mf_":
            listcols=cols2d+cols1d+['cat_id', 'img_id']
        if nsubcases>0: listcols.append('subcase_id')
        alldata.append(imgsmeascats[listcols]) 

    alldata_df=pd.concat(alldata, ignore_index=True)
    alldata_df=alldata_df.sort_values(by=['cat_id',  'img_id'], ignore_index=True)
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)
    logger.info("Grouping finished")
    if len(missing_meas)>0:
        if rot_pair: f=2
        else:f =1
        logger.info("However %i of %i images were not measured, check them"%(len(missing_meas), len(simcats_basename)*f ) )
        print(missing_meas)


#grouping using only realization where there is a good measurement in both galaxies in a pair, measure catalogs must have obj_id
def cats_constimg_rotpair(simdir, measdir, cols2d=["adamom_flux"], cols1d=["tru_s1", "tru_s2"],  filename=None):

    
    '''
    idem but but only include realizations in the catalog where both measurements where good 
    #the obj_id is important for this measure therefore only use this function when adamom measures used tru catalog
    
    '''

    label="adamom_"
    imgdirs_meas_basename = sorted( [ "".join(os.path.basename(f).rsplit("_img", 1)) for f in glob.glob(os.path.join(measdir,'*_img') )] )
    simcats_basename = sorted( [os.path.basename(f).replace("_cat.fits", "") for f in glob.glob(os.path.join(simdir,'*_cat.fits') )] )
    if len(imgdirs_meas_basename) != len( set(imgdirs_meas_basename).intersection(simcats_basename)  ):
        raise RuntimeError("Number of sim catalogs %i does not match number of measurement catalogs %i"%( len(imgdirs_meas_basename), len(simcats_basename) ))

    minobjs=MINOBJS
    alldata = []
    last_obj_id=0
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
     

            if not os.path.exists(meascatpath):
                break
                #raise RuntimeError("Sextractor catalog %s does not exist! Something went wrong when running sextractor before"%( meascatpath ))

            try:
                with fits.open(meascatpath) as hdul:
                    hdul.verify('fix')
                    meascat=hdul[1].data
                #meascat = fitsio.read(meascatpath)
            except Exception as e:
                logger.info("Unable to read existing measurement catalog %s, redoing it"%(meascatpath))
                logger.info(str(e))
                if os.path.exists(meascatpath):
                    logger.info("removing %s"%(meascatpath))
                    #os.remove(meascatpath)
                break

            try:
                with fits.open(meascatpath_rot) as hdul:
                    hdul.verify('fix')
                    meascat_rot=hdul[1].data
                #meascat_rot = fitsio.read(meascatpath_rot)
            except Exception as e:
                logger.info("Unable to read existing measurement catalog %s, redoing it"%(meascatpath_rot))
                logger.info(str(e))
                if os.path.exists(meascatpath_rot):
                    logger.info("removing %s"%(meascatpath_rot))
                    #os.remove(meascatpath_rot)
                break
        
            meascat = meascat.astype(meascat.dtype.newbyteorder('='))
            meascat_df = pd.DataFrame(meascat)
            meascat_rot = meascat_rot.astype(meascat_rot.dtype.newbyteorder('='))
            meascat_rot_df = pd.DataFrame(meascat_rot)

            if "FLAGS" in meascat_df.columns:
                flag=(meascat_df["FLAGS"]==0)&(meascat_df["r_match"]<2.5)
                meascat_df=meascat_df[flag]
                flag_rot=(meascat_rot_df["FLAGS"]==0)&(meascat_rot_df["r_match"]<2.5)
                meascat_rot_df=meascat_rot_df[flag_rot]

            
            #this is required only to include neighbor features
            if len(meascat_df)<minobjs:
                logger.info("Skipping img %s in grouping number of objects < %i"%(meascatpath,minobjs))
                continue

            meascat_df["img_id"] = 2*img_id
            meascat_rot_df["img_id"] = 2*img_id+1
            
            if "gal_density" in cols2d:
                visarea=(4000*0.1/60)**2 #arcmin^2
                #meascat_df["gal_density"] = (np.max(meascat_df["obj_number"])+1)/visarea
                meascat_df["gal_density"] = len(meascat_df)/visarea
                meascat_rot_df["gal_density"] = len(meascat_rot_df)/visarea


            # this lines remove detection bias
            obj_ids=list(set(meascat_df["obj_id"]).intersection(set(meascat_rot_df["obj_id"])))
            meascat_df=meascat_df[meascat_df['obj_id'].isin(obj_ids)]
            meascat_rot_df=meascat_rot_df[meascat_rot_df['obj_id'].isin(obj_ids)]

                        
            #real_obj_id=last_obj_id+np.array(range(len(obj_ids)))
            #last_obj_id=real_obj_id[-1]+1

            c=cantor_pair(cat_id,img_id)
            real_obj_id= np.vectorize(cantor_pair)(c, meascat_df["obj_id"])
            real_obj_id_rot= np.vectorize(cantor_pair)(c, meascat_rot_df["obj_id"])

            #print(meascatpath)
            #print(len(real_obj_id), len(meascat_df), len(obj_ids))

            meascat_df["real_obj_id"]=real_obj_id
            meascat_rot_df["real_obj_id"]=real_obj_id_rot
                
            meascat_dfs.append(meascat_df)
            meascat_dfs.append(meascat_rot_df)
            

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
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)
  

def trucats(simdir, cols1d=['tru_s1','tru_s2'], cols2d=['tru_g1','tru_g2'], filename=None):
    ext='_cat.fits'
    input_catalogs=sorted(glob.glob(os.path.join(simdir, '*%s'%(ext)) ))

    alldata=[]
    for cat_id, catname in enumerate(input_catalogs):
        logger.info("Doing cat %s"%(catname))
        cat = fitsio.read(catname)
        cat= cat.astype(cat.dtype.newbyteorder('='))
        cat = pd.DataFrame(cat)
        cat["cat_id"] = cat_id
        
        trucat_const = fitsio.read(catname, ext=2)
        cols1d=[col for col in cols1d if col in trucat_const.dtype.names]
        cols2d=[col for col in cols2d if col in cat.columns]
        for col in cols1d :
                cat[col] = trucat_const[col][0]
        alldata.append(cat[cols2d+cols1d+['cat_id', 'img_id']])
    alldata_df=pd.concat(alldata, ignore_index=True)
    alldata_df=alldata_df.sort_values(by=['cat_id',  'img_id'], ignore_index=True)
    if filename is None: filename=os.path.join(simdir, 'truegroupcats.fits')
    fitsio.write(filename,  alldata_df.to_records(index=False), clobber=True)

