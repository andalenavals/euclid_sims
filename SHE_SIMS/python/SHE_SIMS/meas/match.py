"""
Add true properties to the measurement via matching with input catalog

"""
import os, glob

import fitsio
from astropy.io import fits
import pandas as pd
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import copy
import datetime
import multiprocessing
from .. import calc

import logging
logger = logging.getLogger(__name__)

def measfct_linear(measdir, inputdir, cols=[], ext='_galimg_meascat.fits', xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="_match", rot_pair=False, ncpu=1):
        
        """
        measdir: directory with the folder containing catalogs of measured properties
        inputdir: directory containing input catalogs (must contain x, y)
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        """
        
        logger.info("Starting matching with input catalog")
        
        detectdirs=sorted(glob.glob(os.path.join(measdir,'*_img') ))      

        for ddir in detectdirs:
                basename=os.path.basename(ddir).replace("_img", "")
                inputcatname=os.path.join(inputdir, '%s%s'%(basename, '_cat.fits'))
                incat=fitsio.read(inputcatname)
                incat= incat.astype(incat.dtype.newbyteorder('='))
                incat = pd.DataFrame(incat)
                nimgs=max(incat["img_id"])+1


                if rot_pair:
                        logger.info("Matching rotated 90 degrees measurements")
                        sncrot=90
                        aux1,aux2=np.vectorize(calc.rotg)(incat["tru_g1"],incat["tru_g2"],sncrot)
                        incat["tru_g1"]=aux1
                        incat["tru_g2"]=aux2
                
                for img_id in range(nimgs):
                        detectcatname=os.path.join(ddir,'%s_img%i%s'%(basename, img_id ,'%s'%(ext)))
                        logger.info("Matching %s"%(detectcatname))
                        try:
                                with fits.open(detectcatname) as hdul:
                                        hdul.verify('fix')
                                        dcat=hdul[1].data
                                        dcat= dcat.astype(dcat.dtype.newbyteorder('='))
                                        dcat = pd.DataFrame(dcat)
                        except Exception as e:
                                logger.info("Failed reading sextractor catalog %s removing it"%(detectcatname))
                                logger.info(str(e))
                                if os.path.isfile(detectcatname):
                                        os.remove(detectcatname)
                                continue

                        cataux=incat[incat['img_id']==img_id]
                        itree=scipy.spatial.KDTree(cataux[['x','y']])
                        distin,indin=itree.query(dcat[[xname,yname]],k=1)
                        dcat["r_match"]=distin
                        for col in cols:
                                dcat["%s%s"%(col,matchlabel)]= cataux[col].to_numpy()[indin]
                                
                        catbintable = fits.BinTableHDU(dcat.to_records(index=False))
                        with fits.open(detectcatname) as hdul:
                                hdul.pop(1)
                                hdul.insert(1, catbintable)
                                hdul.writeto(detectcatname, overwrite=True)


def measfct(measdir, inputdir, cols=[], ext='_galimg_meascat.fits', xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="_match", rot_pair=False, ncpu=1, skipdone=False):
        
        """
        measdir: directory with the folder containing catalogs of measured properties
        inputdir: directory containing input catalogs (must contain x, y)
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        """
        
        logger.info("Starting matching with input catalog")
        wslist=[]
        
        detectdirs=sorted(glob.glob(os.path.join(measdir,'*_img') ))      

        for ddir in detectdirs:
                print(os.path.basename(ddir))
                #basename=os.path.basename(ddir).replace("_img", "")
                basename=os.path.basename(ddir).rsplit("_img", 1)[0]
                inputcatname=os.path.join(inputdir, '%s%s'%(basename, '_cat.fits'))
                incat=fitsio.read(inputcatname)
                incat= incat.astype(incat.dtype.newbyteorder('='))
                incat = pd.DataFrame(incat)
                nimgs=max(incat["img_id"])+1


                if rot_pair:
                        logger.info("Matching rotated 90 degrees measurements")
                        sncrot=90
                        aux1,aux2=np.vectorize(calc.rotg)(incat["tru_g1"],incat["tru_g2"],sncrot)
                        incat["tru_g1"]=aux1
                        incat["tru_g2"]=aux2
                
                for img_id in range(nimgs):
                        detectcatname=os.path.join(ddir,'%s_img%i%s'%(basename, img_id ,'%s'%(ext)))
                        logger.info("Matching %s"%(detectcatname))
                        try:
                                with fits.open(detectcatname) as hdul:
                                        hdul.verify('fix')
                                        dcat=hdul[1].data
                                        dcat= dcat.astype(dcat.dtype.newbyteorder('='))
                                        dcat = pd.DataFrame(dcat)
                        except Exception as e:
                                logger.info("Failed reading sextractor catalog %s removing it"%(detectcatname))
                                logger.info(str(e))
                                if os.path.isfile(detectcatname):
                                        os.remove(detectcatname)
                                continue


                        if skipdone:
                                logger.info("Using skipdone")
                                collab=set(["%s%s"%(c,matchlabel) for c in cols])
                                existingcols=set(dcat.columns).intersection(collab)
                                aux=collab-existingcols
                                if len(aux)==0:
                                        logger.info("All existing input cols were included")
                                        continue
                                else:
                                        logger.info("Measurement file exist but it is missing %s,  %i == %i"%(" ".join(aux), len(collab), len(existingcols)))
                        
                        ws = _WorkerSettings(detectcatname, dcat, incat, img_id, cols, xname, yname,  matchlabel)
                        wslist.append(ws)
        _run(wslist, ncpu)


class _WorkerSettings():
        def __init__(self, detectcatname, dcat, incat, img_id,cols,xname, yname,matchlabel):
                self.detectcatname=detectcatname
                self.dcat= dcat
                self.incat=incat
                self.img_id=img_id
                self.cols=cols
                self.xname=xname
                self.yname=yname
                self.matchlabel=matchlabel
             

def _worker(ws):
        """
        Worker function that the different processes will execute, processing the
        _WorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed() #this is important
        p = multiprocessing.current_process()
        logger.info("%s is starting measure catalog %s with PID %s" % (p.name, str(ws), p.pid))
        
        cataux=ws.incat[ws.incat['img_id']==ws.img_id]
        itree=scipy.spatial.KDTree(cataux[['x','y']])
        distin,indin=itree.query(ws.dcat[[ws.xname,ws.yname]],k=1)
        ws.dcat["r_match"]=distin
        for col in ws.cols:
                if col not in cataux.columns: continue
                ws.dcat["%s%s"%(col,ws.matchlabel)]= cataux[col].to_numpy()[indin]
                                
        catbintable = fits.BinTableHDU(ws.dcat.to_records(index=False))
        with fits.open(ws.detectcatname) as hdul:
                hdul.pop(1)
                hdul.insert(1, catbintable)
                hdul.writeto(ws.detectcatname, overwrite=True)

        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))


def _run(wslist, ncpu):
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
                list(map(_worker, wslist))
        
        else:
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(_worker, wslist)
                pool.close()
                pool.join()
        
        endtime = datetime.datetime.now()
        logger.info("Done, the total running time was %s" % (str(endtime - starttime)))

