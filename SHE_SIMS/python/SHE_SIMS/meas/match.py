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
from ..utils import open_fits, _run

import logging
logger = logging.getLogger(__name__)

def stars(measdir, inputdir, threshold=1.5, ext='_galimg_meascat.fits', xname ='X_IMAGE', yname='Y_IMAGE', ncpu=1, skipdone=False):
        
        """
        measdir: directory with the folder containing catalogs of measured properties
        inputdir: directory containing input catalogs (must contain x, y)
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        threshold: maximum allowed divergence in the matching of catalogs
        """
        
        logger.info("Starting matching with input catalog")
        
        detectdirs=sorted(glob.glob(os.path.join(measdir,'*_img') ))      
        wslist=[]
        for ddir in detectdirs:
                basename=os.path.basename(ddir).replace("_img", "")
                inputcatname=os.path.join(inputdir, '%s%s'%(basename, '_cat.fits'))
                incat=open_fits(inputcatname, hdu=3)
                nimgs=max(incat["img_id"])+1

                for img_id in range(nimgs):
                        detectcatname=os.path.join(ddir,'%s_img%i%s'%(basename, img_id ,'%s'%(ext)))
                        logger.info("Matching %s"%(detectcatname))
                        dcat=open_fits(detectcatname)
                        '''
                        if skipdone:
                                logger.info("Using skipdone")
                                collab=set(["r_star", "star_flag"])
                                existingcols=set(dcat.columns).intersection(collab)
                                aux=collab-existingcols
                                if len(aux)==0:
                                        logger.info("All existing input cols were included")
                                        continue
                                else:
                                        logger.info("Measurement file exist but it is missing %s,  %i == %i"%(" ".join(aux), len(collab), len(existingcols)))
                        '''
                        ws = _StarWorkerSettings(detectcatname, dcat, incat, img_id, xname, yname,threshold)
                        wslist.append(ws)
        _run(_starworker, wslist, ncpu)
                        
                        
class _StarWorkerSettings():
        def __init__(self, detectcatname, dcat, incat, img_id,xname, yname, threshold):
                self.detectcatname=detectcatname
                self.dcat= dcat
                self.incat=incat
                self.img_id=img_id
                self.xname=xname
                self.yname=yname
                self.threshold=threshold
             

def _starworker(ws):
        from ..variables import GALCOLS
        """
        Worker function that the different processes will execute, processing the
        _WorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed() #this is important
        p = multiprocessing.current_process()
        logger.info("%s is starting measure catalog %s with PID %s" % (p.name, str(ws.detectcatname), p.pid))
        
        cataux=ws.incat[ws.incat['img_id']==ws.img_id]
        itree=scipy.spatial.KDTree(cataux[['x','y']])
        distin,indin=itree.query(ws.dcat[[ws.xname,ws.yname]],k=1)
        ws.dcat["r_star"]=distin
        ws.dcat["star_flag"]=0
        useflag=(ws.dcat["r_star"]<=ws.threshold)
        if "r_match" in ws.dcat.columns:
                useflag=(ws.dcat["r_match"]>ws.dcat["r_star"]) #that is more accurate than setting a threshold
        ws.dcat.loc[useflag, "star_flag"]=1
   
        for f in GALCOLS+["tru_g1","tru_g2"]:
                if f not in list(ws.dcat.columns): continue
                if f=="dominant_shape":continue
                ws.dcat.loc[useflag, f]=[-999]*int(np.sum(useflag))
        
        for f in ["x","y","tru_flux", "tru_mag"]:
                if f not in ws.dcat.columns: continue
                target_dtype=ws.dcat[f].dtype
                ws.dcat.loc[useflag, f]=cataux[f].to_numpy()[indin][useflag].astype(target_dtype)

        catbintable = fits.BinTableHDU(ws.dcat.to_records(index=False))
        with fits.open(ws.detectcatname) as hdul:
                hdul.pop(1)
                hdul.insert(1, catbintable)
                hdul.writeto(ws.detectcatname, overwrite=True)
                                
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))


                                
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
                incat=open_fits(inputcatname)
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
                        dcat=open_fits(detectcatname)
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
        _run(_worker, wslist, ncpu)


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
        logger.info("%s is starting measure catalog %s with PID %s" % (p.name, str(ws.detectcatname), p.pid))
        
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




