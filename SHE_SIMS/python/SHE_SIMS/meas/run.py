"""
High-level functionality to measure features on several images using multiprocessing.

We have two public functions:

- a onsims() which is specialised in exploring the file structure made by
  momentsml.sim.run.multi(), and run on them by calling
- general(), which can be used for any images.

The approach taken by general() is the following:

- make a list of _WorkerSetting objects, each of them describing the elementary task of
  measuring shapes on one single image.
- feed this list into _run(), which takes care of running the measurements.
  We use the simple multiprocessing.Pool to distribute the work on several cpus.
  This seems easier than implementing a pool-like queue ourself.

  
"""


import os
import sys
import glob
import datetime
import multiprocessing
import copy
import os
import fitsio
import astropy.io.fits as fits
import warnings
warnings.filterwarnings("error")
import astropy.io.fits
from astropy.io import fits
import numpy as np
import galsim

from . import sex
from . import galsim_adamom
from . import galsim_ksb
from . import ngmix
from ..utils import _run

import logging
logger = logging.getLogger(__name__)

###########################
### RUN KSB on SIM ###
###########################

def makeksbworkerlist(simdir, measdir, sexdir, measkwargs, ext='_galimg', catsdir=None):
        wslist = []
        if measkwargs["cattype"]=='sex':
                logger.info("Using sextractor catalog (positions) for getting galsim_KSB")
                if sexdir is None:
                        raise
                logger.info("Exploring the contents of simdir '%s'" % (simdir))
                imgfolders = sorted(glob.glob(os.path.join(simdir, "*_img")))
                for folder in imgfolders:
                        #catfilename=folder.replace("_img", "_cat.fits")
                        catfilename="_cat.fits".join(folder.rsplit("_img",1))
                        if catsdir is not None:
                                catfilename=os.path.join(catsdir,os.path.basename(catfilename))
                        print(catfilename)
                        assert os.path.isfile(catfilename)
                        constcat=fitsio.read(catfilename, ext=2)
                        nimgs=constcat['nimgs'][0]
                        tru_type=constcat['tru_type'][0]
                        psffile=os.path.join(str(constcat['psf_path'][0]),str(constcat['psf_file'][0]))
                        
                        name=os.path.basename(folder) 
                        imgdir_meas=os.path.join(measdir, name )
                        if not os.path.exists(imgdir_meas):
                                os.makedirs(imgdir_meas)
                                logger.info("Creating a new set of measurementes named '%s'" % (imgdir_meas))
                        else:
                                logger.info("Adding new measurementes to the existing set '%s'" % (imgdir_meas))

                        
                        imgs=sorted(glob.glob(os.path.join(folder, "*%s.fits"%(ext))))
                        logger.info("Case have %i images in this case"%(len(imgs)))

                        for img in imgs:
                                logger.info("Doing image %s"%(img))

                                sex_cat_name=os.path.join(sexdir, name, os.path.basename(img).replace("%s.fits"%(ext),"%s_cat.fits"%(ext)))
                                
                                if  os.path.isfile(sex_cat_name):
                                     if not os.path.getsize(sex_cat_name):
                                             #os.remove(sex_cat_name)
                                             logger.warning("Corrupted sextractor catalog %s"%(sex_cat_name))
                                             break
                                else:
                                        logger.warning("Could not find sextractor catalog %s"%(sex_cat_name))
                                        break
                                        #continue
                                

                                weight=os.path.join(sexdir, name, os.path.basename(img).replace("%s.fits"%(ext),"%s_seg.fits"%(ext)))
                                if not os.path.exists(weight):
                                        logger.warning("Segmentation maps %s does not exist"%(weight))
                                        weight=None
                                if (weight is None) & (measkwargs["use_weight"]):
                                        logger.error("You want to measure with weights but there is not segmentation map avaliable")
                                        break

                                
                                outcat = os.path.join(imgdir_meas,os.path.basename(img).replace("%s.fits"%(ext),"%s_meascat.fits"%(ext)))
                                if measkwargs["skipdone"]:
                                        if os.path.isfile(outcat):
                                                try:
                                                        with fits.open(outcat) as hdul:
                                                                hdul.verify('fix')
                                                                meascat=hdul[1].data
                                                                #meascat=fitsio.read(filename,ext=2)
                                                                
                                                        if measkwargs["extra_cols"] is not None:
                                                                li=len(set(meascat.dtype.names).intersection(set(measkwargs["extra_cols"])))
                                                                if li==len(measkwargs["extra_cols"]):
                                                                        logger.info("Measurement file exist and have all existing extra column from sex cat")
                                                                        continue
                                                        else:
                                                                logger.info("Measurement file exist and there are not extra cols")
                                                                continue
                                                except Exception as e:
                                                        logger.info("Unable to read existing measurement catalog %s, redoing it"%(outcat))
                                                        logger.info(str(e))
                                                        
                                ws = _KSBWorkerSettings(img, psffile, sex_cat_name, None,  weight,  outcat, measkwargs)
                                wslist.append(ws)

                        logger.info("Using %i / %i images"%(len(imgs),len(wslist) ) )

                        
        else:        
                logger.info("Exploring the contents of simdir '%s'" % (simdir))
        
                incats = sorted(glob.glob(os.path.join(simdir, "*_cat.fits")))
                
                for incat in incats:
                        constcat=fitsio.read(incat, ext=2)
                        nimgs=constcat['nimgs'][0]
                        tru_type=constcat['tru_type'][0]
                        psffile=os.path.join(str(constcat['psf_path'][0]),str(constcat['psf_file'][0]))

                        name = os.path.basename(incat).replace("_cat.fits", "")
                        imgdir=os.path.join(simdir,"%s_img"%(name))
                        imgdir_meas=os.path.join(measdir,"%s_img"%(name))
                        if not os.path.exists(imgdir_meas):
                                os.makedirs(imgdir_meas)
                                logger.info("Creating a new set of measurementes named '%s'" % (imgdir_meas))
                        else:
                                logger.info("Adding new measurementes to the existing set '%s'" % (imgdir_meas))

                        # And loop over the "declared" realization ImageInfo objects:
                        for img_id in range(nimgs):
                                imgname=os.path.join(imgdir,"%s_img%i%s.fits"%(name,img_id,ext))

                                # Let's check that the declared image file does exist:
                                if not os.path.exists(imgname):
                                        logger.error("Could not find image realization '%s', will skip it" % (imgname))
                                        break
                                        
                                outcat = os.path.join(imgdir_meas,"%s_img%i%s_meascat.fits"%(name,img_id,ext))
                                if measkwargs["skipdone"]:
                                        if os.path.isfile(outcat):
                                                try:
                                                        with fits.open(outcat) as hdul:
                                                                hdul.verify('fix')
                                                                meascat=hdul[1].data
                                                                #meascat=fitsio.read(filename,ext=2)
                                                        if measkwargs["extra_cols"] is not None:
                                                                li=len(set(meascat.dtype.names).intersection(set(measkwargs["extra_cols"])))
                                                                if li==len(measkwargs["extra_cols"]):
                                                                        logger.info("Measurement file exist and have all existing extra column from sex cat")
                                                                        continue
                                                        else:
                                                                logger.info("Measurement file exist and there are not extra cols")
                                                                continue
                                                except Exception as e:
                                                        logger.info("Unable to read existing measurement catalog %s, redoing it"%(outcat))
                                                        logger.info(str(e))
                                        
                                weight=os.path.join(sexdir, "%s_img"%(name), os.path.basename(imgname).replace("%s.fits"%(ext),"%s_seg.fits"%(ext)))
                                if not os.path.exists(weight):
                                        logger.warning("Segmentation maps %s does not exist"%(weight))
                                        weight=None
                                if (weight is None) & (measkwargs["use_weight"]):
                                        logger.error("You want to measure with weights but there is not segmentation map avaliable")
                                        break

                                ws = _KSBWorkerSettings(imgname, psffile, incat, img_id, weight,  outcat, measkwargs)
                                wslist.append(ws)
        return wslist
        
def ksb(simdir, measdir, measkwargs, sexdir=None,catsdir=None,  ncpu=1, rot_pair=True):
        '''
        simdir: work dir containing folder with _cat.fits and _img/. Each pair (_cat.fits, _img) correspond to a case. img* folder contains all realizations
        measdir: path where galsim_adamom measures will be saved
        seaxdir: directory with sextractor measures (for position and segmentation maps)
        use_weight: use segmentation maps as weight function 
        rot_pair: measure the rot_pair images avaliable
        '''

        
        if not os.path.exists(measdir):
                os.makedirs(measdir)
        wslist=makeksbworkerlist(simdir, measdir, sexdir, measkwargs, ext='_galimg', catsdir=catsdir)
        '''
        chunksize=1*ncpu
        nchunks=len(wslist)//chunksize
        logger.info("CHUNKSIZE: %i"%(chunksize))
        logger.info("NCHUNKS: %i"%(nchunks))
        for i in range(nchunks):
                idi=i*chunksize
                idf=(i+1)*chunksize
                if i==nchunks-1: idf=None
                _run(wslist[idi:idf], ncpu)
        if nchunks==0:
                _run(wslist, ncpu)
        '''
        _run(_ksbworker, wslist, ncpu)
        if rot_pair:
                measkwargs.update({"rot_pair":rot_pair})
                wslist=makeksbworkerlist(simdir, measdir, sexdir, measkwargs, ext='_galimg_rot', catsdir=catsdir)     
                _run(_ksbworker, wslist, ncpu)
        

class _KSBWorkerSettings():
        """
        #A class that holds together all the settings for measuring an image.
        """
        def __init__(self, imgname, psffile, cat_img, img_id, weight, filename, measkwargs):
                
                self.imgname = imgname
                self.psffile = psffile
                self.cat_img = cat_img
                self.img_id = img_id
                self.weight = weight
                self.filename = filename
                self.measkwargs=measkwargs

def _ksbworker(ws):
        """
        #Worker function that the different processes will execute, processing the
        #_WorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        
        p = multiprocessing.current_process()
        logger.info("%s is starting to measure catalog %s with PID %s" % (p.name, str(ws.cat_img), p.pid))

        
        galsim_ksb.measure(ws.imgname,  ws.psffile, ws.cat_img, ws.img_id, weight=ws.weight, filename=ws.filename,  **ws.measkwargs)

        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))


###########################
### RUN ADAMOM on SIM ###
###########################

def makeworkerlist(simdir, measdir, sexdir, cattype,skipdone, measkwargs, ext='_galimg'):
        wslist = []
        if cattype=='sex':
                logger.info("Using sextractor catalog (positions) for getting galsim_adamom")
                if sexdir is None:
                        raise
                logger.info("Exploring the contents of simdir '%s'" % (simdir))
                imgfolders = sorted(glob.glob(os.path.join(simdir, "*_img")))
                for folder in imgfolders:
                        name=os.path.basename(folder) 
                        imgdir_meas=os.path.join(measdir, name )
                        if not os.path.exists(imgdir_meas):
                                os.makedirs(imgdir_meas)
                                logger.info("Creating a new set of measurementes named '%s'" % (imgdir_meas))
                        else:
                                logger.info("Adding new measurementes to the existing set '%s'" % (imgdir_meas))

                        
                        imgs=sorted(glob.glob(os.path.join(folder, "*%s.fits"%(ext))))
                        logger.info("Case have %i images in this case"%(len(imgs)))

                        for img in imgs:
                                logger.info("Doing image %s"%(img))

                                sex_cat_name=os.path.join(sexdir, name, os.path.basename(img).replace("%s.fits"%(ext),"%s_cat.fits"%(ext)))
                                
                                if  os.path.isfile(sex_cat_name):
                                     if not os.path.getsize(sex_cat_name):
                                             #os.remove(sex_cat_name)
                                             logger.warning("Corrupted sextractor catalog %s"%(sex_cat_name))
                                             break
                                else:
                                        logger.warning("Could not find sextractor catalog %s"%(sex_cat_name))
                                        break
                                        #continue
                                

                                weight=os.path.join(sexdir, name, os.path.basename(img).replace("%s.fits"%(ext),"%s_seg.fits"%(ext)))
                                if not os.path.exists(weight):
                                        logger.debug("Segmentation maps %s does not exist"%(weight))
                                        weight=None

                                
                                outcat = os.path.join(imgdir_meas,os.path.basename(img).replace("%s.fits"%(ext),"%s_meascat.fits"%(ext)))
                                if skipdone:
                                        extra_cols=measkwargs["extra_cols"]
                                        logger.info("Using skipdone")
                                        if os.path.isfile(outcat):
                                                logger.info("%s is a file. Evaluating ..."%(outcat))
                                                try:
                                                        with fits.open(outcat) as hdul:
                                                                hdul.verify('fix')
                                                                meascat=hdul[1].data
                                                        with fits.open(sex_cat_name) as hdul:
                                                                hdul.verify('fix')
                                                                sexcat=hdul[2].data
                                                        logger.info("Read sucessfully!")

                                                        if extra_cols is not None:
                                                                extracols=set(sexcat.dtype.names).intersection(set(extra_cols))
                                                                li=set(meascat.dtype.names).intersection(set(extra_cols))
                                                                aux=extracols-li
                                                                if len(aux)==0:
                                                                        logger.info("Measurement file exist and have all existing extra column from sex cat")
                                                                        continue
                                                                else:
                                                                        logger.info("Measurement file exist but it is missing %s,  %i == %i"%(" ".join(aux), len(extracols), len(li)))
                                                        else:
                                                                logger.info("Measurement file exist and there are not extra cols")
                                                                continue
                                                except Exception as e:
                                                        logger.info("Unable to read existing measurement catalog %s, redoing it"%(outcat))
                                                        logger.info(str(e))
                                        else:
                                                logger.info("%s is not a file. You are measuring for the first time"%(outcat))
                                                        
                                ws = _AdamomWorkerSettings(img, sex_cat_name, None, weight,  outcat, cattype, measkwargs)
                                wslist.append(ws)

                        logger.info("Using %i / %i images"%(len(imgs),len(wslist) ) )

                        
        else:        
                logger.info("Exploring the contents of simdir '%s'" % (simdir))
        
                incats = sorted(glob.glob(os.path.join(simdir, "*_cat.fits")))
                
                for incat in incats:
                        constcat=fitsio.read(incat, ext=2)
                        nimgs=constcat['nimgs'][0]
                        tru_type=constcat['tru_type'][0]
                        

                        name = os.path.basename(incat).replace("_cat.fits", "")
                        imgdir=os.path.join(simdir,"%s_img"%(name))
                        imgdir_meas=os.path.join(measdir,"%s_img"%(name))
                        if not os.path.exists(imgdir_meas):
                                os.makedirs(imgdir_meas)
                                logger.info("Creating a new set of measurementes named '%s'" % (imgdir_meas))
                        else:
                                logger.info("Adding new measurementes to the existing set '%s'" % (imgdir_meas))

                        # And loop over the "declared" realization ImageInfo objects:
                        for img_id in range(nimgs):
                                imgname=os.path.join(imgdir,"%s_img%i%s.fits"%(name,img_id,ext))

                                # Let's check that the declared image file does exist:
                                if not os.path.exists(imgname):
                                        logger.error("Could not find image realization '%s', will skip it" % (imgname))
                                        break
                                        
                                outcat = os.path.join(imgdir_meas,"%s_img%i%s_meascat.fits"%(name,img_id,ext))
                                if skipdone:
                                        extra_cols=measkwargs["extra_cols"]
                                        if os.path.isfile(outcat):
                                                try:
                                                        with fits.open(outcat) as hdul:
                                                                hdul.verify('fix')
                                                                meascat=hdul[1].data
                                                                #meascat=fitsio.read(filename,ext=2)
                                                        with fits.open(incat) as hdul:
                                                                hdul.verify('fix')
                                                                inames=hdul[1].data.dtype.names
                                                                #inames2=hdul[2].data.dtype.names
                                                                #inames=inames1+inames2
                                                        if extra_cols is not None:
                                                                extracols=set(inames).intersection(set(extra_cols))
                                                                li=set(meascat.dtype.names).intersection(set(extra_cols))
                                                                aux=extracols-li
                                                                if len(aux)==0:
                                                                        logger.info("Measurement file exist and have all existing extra column from input cat")
                                                                        continue
                                                                else:
                                                                        logger.info("Measurement file exist but it is missing %s,  %i == %i"%(" ".join(aux), len(extracols), len(li)))
                                                        else:
                                                                logger.info("Measurement file exist and there are not extra cols")
                                                                continue
                                                except Exception as e:
                                                        logger.info("Unable to read existing measurement catalog %s, redoing it"%(outcat))
                                                        logger.info(str(e))
                                        
                                weight=os.path.join(sexdir, "%s_img"%(name), os.path.basename(imgname).replace("%s.fits"%(ext),"%s_seg.fits"%(ext)))
                                if not os.path.exists(weight):
                                        logger.debug("Segmentation maps %s does not exist"%(weight))
                                        weight=None

                                ws = _AdamomWorkerSettings(imgname, incat, img_id, weight,  outcat, cattype, measkwargs)
                                wslist.append(ws)
        return wslist
        
def adamom(simdir, measdir, measkwargs,  sexdir=None, cattype='tru', ncpu=1, skipdone=True, rot_pair=True):
        '''
        simdir: work dir containing folder with _cat.fits and _img/. Each pair (_cat.fits, _img) correspond to a case. img* folder contains all realizations
        measdir: path where galsim_adamom measures will be saved
        seaxdir: directory with sextractor measures (for position and segmentation maps)
        cattype: type of catalog for initial position for fitting
        use_weight: use segmentation maps as weight function 
        rot_pair: measure the rot_pair images avaliable
        '''

        
        if not os.path.exists(measdir):
                os.makedirs(measdir)
        wslist=makeworkerlist(simdir, measdir, sexdir, cattype,  skipdone,  measkwargs, ext='_galimg')
        '''
        chunksize=1*ncpu
        nchunks=len(wslist)//chunksize
        logger.info("CHUNKSIZE: %i"%(chunksize))
        logger.info("NCHUNKS: %i"%(nchunks))
        for i in range(nchunks):
                idi=i*chunksize
                idf=(i+1)*chunksize
                if i==nchunks-1: idf=None
                _run(wslist[idi:idf], ncpu)
        if nchunks==0:
                _run(wslist, ncpu)
        '''
        _run(_worker, wslist, ncpu)
        if rot_pair:
                measkwargs.update({"rot_pair":rot_pair})
                wslist=makeworkerlist(simdir, measdir, sexdir, cattype,  skipdone,  measkwargs,  ext='_galimg_rot')     
                _run(_worker, wslist, ncpu)
        

class _AdamomWorkerSettings():
        """
        #A class that holds together all the settings for measuring an image.
        """
        
        def __init__(self, imgname, cat_img, img_id, weight, filename,cat_type, measkwargs):
                
                self.imgname = imgname
                self.cat_img = cat_img
                self.img_id = img_id
                self.weight = weight
                self.filename = filename
                self.cat_type = cat_type
                self.measkwargs= measkwargs

def _worker(ws):
        """
        #Worker function that the different processes will execute, processing the
        #_WorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        
        p = multiprocessing.current_process()
        logger.info("%s is starting to measure catalog %s with PID %s" % (p.name, str(ws.cat_img), p.pid))

        
        if ws.cat_type=='sex':     
                galsim_adamom.measure_withsex(ws.imgname, ws.cat_img, weight=ws.weight, filename=ws.filename, **ws.measkwargs )
        else:
                galsim_adamom.measure(ws.imgname, ws.cat_img, ws.img_id, weight=ws.weight, filename=ws.filename, **ws.measkwargs)

        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))



###########################
### RUN NGMIX on SIM ###
##########################
def makengmixworkerlist(simdir, measdir,  measkwargs, sexdir=None, cattype='tru', ncpu=1, skipdone=True, ext='_galimg', catsdir=None):
        
        #simdir: work dir containing folder with _cat.fits and _img/. Each pair (_cat.fits, _img) correspond to a case. img* folder contains all realizations
        #measdir: path where galsim_adamom measures will be saved
        #seaxdir: directory with sextractor measures (for position and segmentation maps)
        #cattype: type of catalog for initial position for fitting
        #use_weight: use segmentation maps as weight function 
        if not os.path.exists(measdir):
                os.makedirs(measdir)
        wslist = []
        
        if cattype=='sex':
                logger.info("Using sextractor catalog (positions) for getting galsim_adamom")
                if sexdir is None:
                        raise
                logger.info("Exploring the contents of simdir '%s'" % (simdir))
                imgfolders = sorted(glob.glob(os.path.join(simdir, "*_img")))
                for folder in imgfolders:
                        #catfilename=os.path.join(os.path.dirname(folder), "%s_cat.fits"%(os.path.basename(folder)[:-4]))
                        catfilename="_cat.fits".join(folder.rsplit("_img",1))
                        if catsdir is not None:
                                catfilename=os.path.join(catsdir,os.path.basename(catfilename))
                        assert os.path.isfile(catfilename)
                        constcat=fitsio.read(catfilename, ext=2)
                        nimgs=constcat['nimgs'][0]
                        tru_type=constcat['tru_type'][0]
                        psffile=str(constcat['psf_file'][0])

                        name=os.path.basename(folder) 
                        imgdir_meas=os.path.join(measdir, name )
                        if not os.path.exists(imgdir_meas):
                                os.makedirs(imgdir_meas)
                                logger.info("Creating a new set of measurementes named '%s'" % (imgdir_meas))
                        else:
                                logger.info("Adding new measurementes to the existing set '%s'" % (imgdir_meas))

                        
                        imgs=sorted(glob.glob(os.path.join(folder, "*%s.fits")%(ext) ))
                        logger.info("Case have %i images in this case"%(len(imgs)))

                        for img in imgs:
                                logger.info("Doing image %s"%(img))

                                sex_cat_name=os.path.join(sexdir, name, os.path.basename(img).replace("%s.fits"%(ext),"%s_cat.fits"%(ext)))

                                assert os.path.isfile(sex_cat_name)

                                try:
                                        sex_cat=fitsio.read(sex_cat_name,ext=2)
                                except:
                                        logger.info("Unable to read catalog %s"%(sex_cat_name))
                                        continue
      
                                if len(sex_cat)==0:
                                        logger.info("Skipping: empty sextractor catalog %s"%(sex_cat_name))
                                        
                                
                                if not os.path.exists(sex_cat_name):
                                        logger.warning("Could not find sextractor catalog %s, for image %s " % (sex_cat, img))
                                        continue

                                weight=os.path.join(sexdir, name, os.path.basename(img).replace("%s.fits"%(ext),"%s_seg.fits"%(ext)))
                                if not os.path.exists(weight):
                                        logger.warning("Segmentation maps %s does not exist"%(weight))
                                        weight=None

                                
                                outcat = os.path.join(imgdir_meas,os.path.basename(img).replace("%s.fits"%(ext),"%s_meascat.fits"%(ext)))

                                if skipdone and os.path.exists(outcat):
                                        logger.info("Output catalog %s already exists, skipping this one..." % (outcat))        
                                        continue
                                measkwargs.update({"weight":weight, "filename":outcat,"xname":"X_IMAGE","yname":"Y_IMAGE" })
                                ws = _NgmixWorkerSettings(img,psffile, sex_cat, copy.deepcopy(measkwargs) )
                                wslist.append(ws)

                        logger.info("Using %i / %i images"%(len(imgs),len(wslist) ) )

                        
        else:        
                logger.info("Exploring the contents of simdir '%s'" % (simdir))
        
                incats = sorted(glob.glob(os.path.join(simdir, "*_cat.fits")))
                
                for incat in incats:
                        cat = fitsio.read(incat)

                        name = os.path.basename(incat).replace("_cat.fits", "")
                        imgdir=os.path.join(simdir,"%s_img"%(name))
                        imgdir_meas=os.path.join(measdir,"%s_img"%(name))
                        if not os.path.exists(imgdir_meas):
                                os.makedirs(imgdir_meas)
                                logger.info("Creating a new set of measurementes named '%s'" % (imgdir_meas))
                        else:
                                logger.info("Adding new measurementes to the existing set '%s'" % (imgdir_meas))

                        # And loop over the "declared" realization ImageInfo objects:
                        for img_id in range(max(cat["img_id"]) +1 ):
                                imgname=os.path.join(imgdir,"%s_img%i%s.fits"%(name,img_id,ext))

                                # Let's check that the declared image file does exist:
                                if not os.path.exists(imgname):
                                        logger.warning("Could not find image realization '%s', will skip it" % (imgname))
                                        continue
                                
                                outcat = os.path.join(imgdir_meas,"%s_img%i%s_meascat.fits"%(name,img_id,ext))
                                if skipdone and os.path.exists(outcat):
                                        logger.info("Output catalog %s already exists, skipping this one..." % (outcat))        
                                        continue
                        
                                trucat_img = cat[cat["img_id"]==img_id]
                                        
                                weight=os.path.join(sexdir, "%s_img"%(name), os.path.basename(imgname).replace("%s.fits"%(ext),"%s_seg.fits"%(ext)))
                                if not os.path.exists(weight):
                                        logger.warning("Segmentation maps %s does not exist"%(weight))
                                        weight=None
                                measkwargs.update({"weight":weight, "filename":outcat})
                                ws = _NgmixWorkerSettings(imgname, trucat_img,  measkwargs )
                                wslist.append(ws)
        return wslist
        
def ngmix_meas(simdir, measdir, measkwargs,  sexdir=None, catsdir=None,cattype='tru', ncpu=1, skipdone=True, rot_pair=True):
        '''
        simdir: work dir containing folder with _cat.fits and _img/. Each pair (_cat.fits, _img) correspond to a case. img* folder contains all realizations
        measdir: path where galsim_adamom measures will be saved
        seaxdir: directory with sextractor measures (for position and segmentation maps)
        cattype: type of catalog for initial position for fitting
        use_weight: use segmentation maps as weight function 
        rot_pair: measure the rot_pair images avaliable
        '''

        
        if not os.path.exists(measdir):
                os.makedirs(measdir)
        wslist=makengmixworkerlist(simdir, measdir, measkwargs, sexdir, cattype,  skipdone,  ext='_galimg', catsdir=catsdir)
        _run(_ngmixworker, wslist, ncpu)
        if rot_pair:
                measkwargs.update({"rot_pair":rot_pair})
                wslist=makengmixworkerlist(simdir, measdir, measkwargs, sexdir, cattype,  skipdone,  ext='_galimg_rot', catsdir=catsdir)     
                _run(_ngmixworker, wslist, ncpu)
  
class _NgmixWorkerSettings():
        """
        A class that holds together all the settings for measuring an image.
        """                
        def __init__(self, imgname, psfname, cat, measkwargs):
                
                self.imgname = imgname
                self.psfname = psfname
                self.cat = cat
                self.measkwargs= measkwargs
        
def _ngmixworker(ws):
        """
        #Worker function that the different processes will execute, processing the
        #_WorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        
        p = multiprocessing.current_process()
        logger.info("%s is starting to measure %s with PID %s" % (p.name, str(ws.imgname), p.pid))

        
        ngmix.measure(ws.imgname, ws.psfname, ws.cat, **ws.measkwargs)

        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))

   
        
        
###########################
# RUN SEXTRACTOR++ ON SIM #
###########################
def sextractorpp(simdir, sex_bin=None, sex_config=None, sex_params=None, sex_filter=None, python_config=None, run_check=True, use_psfimg=True, psf_file=None, skipdone=False, ncpu=1, nthreads=1, ext='_galimg.fits', measdir=None, catext="_cat.fits", checkext='_seg.fits', strongcheck=False):
        if measdir is None:
                measdir="%s"%(simdir)
        images_folders = sorted(glob.glob(os.path.join(simdir,'*_img') ))
        logger.info('The work dir have %i images'%(len(images_folders)))

        wslist = []
        psfext='_psfcoreimg.fits'
        for img_path in images_folders:
                filenames = sorted(glob.glob(os.path.join(img_path,'*%s'%(ext)) ))
                if (len(filenames) == 0):
                        logger.info("Skipping dir %s not %s files"%(img_path,ext))
                        continue
                #assert len(filenames) ==1
                if sex_params is not None:
                        sex_pars = "--output-properties %s"%(sex.read_sexparam(sex_params))
                else:
                        sex_pars = ""

                name = os.path.basename(img_path).replace("_img", "")
                imgdir_meas=os.path.join(measdir,"%s_img"%(name))
                if not os.path.exists(imgdir_meas):
                        os.makedirs(imgdir_meas)
                        logger.info("Creating a new set of sextractor measurementes named '%s'" % (imgdir_meas))
                else:
                        logger.info("Adding new sextractor measurementes to the existing set '%s'" % (imgdir_meas))
                        
                for img_file in filenames:
                
                        if ( not os.path.isfile(img_file)):
                                logger.info("Warning %s does not exist skipping sextractor run"%(image_file)) 
                                continue
                        
                        base = os.path.splitext(img_file)[0].replace(simdir, measdir)
                        cat_file = '%s%s'%(base,catext)
                        if run_check:
                                #check_path=os.path.join(os.path.dirname(img_file), os.path.basename(img_file).replace(".fits",""))
                                #check_flags = sex.get_checkflags(check_path)
                                check_file = '%s%s'%(base,checkext)
                                check_flags= '--check-image-segmentation %s'%(check_file) 
                        else:
                                check_flags = ""
                                
                        if skipdone:
                                if ( os.path.isfile(cat_file)):
                                        logger.info("%s exist sextractor run checking it"%(cat_file))
                                        
                                        try:
                                                if strongcheck:
                                                        if check_file is not None:
                                                                with fits.open(check_file) as hdul:
                                                                        hdul.verify('fix')
                                                                        cat_fitsio=hdul[1].data
                                                                #cat_fitsio=fitsio.read(check_file)
                                                        with fits.open(cat_file) as hdul:
                                                                hdul.verify('fix')
                                                                cat_fitsio=hdul[1].data
                                                        
                                                        #cat_fitsio=fitsio.read(cat_file)
                                                        logger.info("%s It is OK!!"%(cat_file))
                                                        continue
                                                else:
                                                        if check_file is not None:
                                                                assert (os.path.isfile(check_file))
                                                                assert bool(os.path.getsize(check_file))
                                                        assert (os.path.isfile(cat_file))
                                                        assert bool(os.path.getsize(cat_file))
                                                        logger.info("%s It is OK!!"%(cat_file))
                                                        continue
                                        except:
                                                logger.info("Something went wrong with image %s"%(check_file))
                                                logger.info("Trying measuring again")
                                                '''
                                                if check_file is not None:
                                                        if os.path.isfile(check_file):
                                                                os.remove(check_file)
                                                if os.path.isfile(cat_file):
                                                        os.remove(cat_file)
                                                '''
                        
                        if use_psfimg:
                                if psf_file is None:
                                        psf_file=img_file.replace(ext,psfext)
                                        if ( not os.path.isfile(psf_file)):
                                                logger.info("Warning %s does not exist skipping sextractor run"%(psf_file)) 
                                                continue
                        else:
                                psf_file= None
                
                        ws = _SexppWorkerSettings(img_file, cat_file,sex_filter, sex_bin, sex_config, sex_pars, python_config, psf_file, check_flags,nthreads )
                        wslist.append(ws)
        
        logger.info("Ready to run Sextractor measurements on %i images." % (len(wslist)))
        _run(_sexppworker, wslist, ncpu)
           

class _SexppWorkerSettings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, img_file, cat_file, sex_filter, sex_bin, sex_config, sex_params, python_config, psf_file, check_flags, nthreads):
                
                self.img_file= img_file
                self.cat_file = cat_file
                self.sex_filter = sex_filter
                self.sex_bin = sex_bin
                self.sex_config = sex_config
                self.sex_params = sex_params
                self.python_config = python_config
                self.psf_file=psf_file
                self.check_flags=check_flags
                self.nthreads=nthreads
                

def _sexppworker(ws):
        """
        Worker function that the different processes will execute, processing the
        _SexWorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        p = multiprocessing.current_process()
        logger.info("%s is starting Sextractor measure catalog %s with PID %s" % (p.name, str(ws), p.pid))
        
        sex.run_SEXTRACTORPP(ws.img_file, ws.cat_file, ws.sex_filter,
                             ws.sex_bin, ws.sex_config, ws.sex_params,
                             ws.python_config, ws.psf_file,
                             ws.check_flags, ws.nthreads, logger)

        
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))

   
###########################
# RUN SEXTRACTOR ON SIM #
###########################
def sextractor(simdir, sex_bin, sex_config, sex_params, sex_filter, sex_nnw, skipdone=False, measdir=None, ext='_galimg.fits', catext="_cat.fits", checkext='_seg.fits', ncpu=1, run_check=False, strongcheck=False):
        # strongcheck: when skipdone open all the expected outputs and skip if succesful
        
        if measdir is None:
                measdir="%s"%(simdir)
                
        images_folders = sorted(glob.glob(os.path.join(simdir,'*_img') ))
        logger.info('The work dir have %i images'%(len(images_folders)))

        wslist = []
        
        for img_path in images_folders:
                filenames = sorted(glob.glob(os.path.join(img_path,'*%s'%(ext)) ))
                filenames += sorted(glob.glob(os.path.join(img_path,'*_galimg_rot.fits') ))
                if (len(filenames) == 0):
                        logger.info("Skipping dir %s not %s files"%(img_path,ext))
                        continue

                #making measurmen dirs                
                imgdir_meas=os.path.join(measdir,os.path.basename(img_path))
                if not os.path.exists(imgdir_meas):
                        os.makedirs(imgdir_meas)
                        logger.info("Creating a new set of sextractor measurementes named '%s'" % (imgdir_meas))
                else:
                        logger.info("Adding new sextractor measurementes to the existing set '%s'" % (imgdir_meas))

                        
                for img_file in filenames:
                
                        base = os.path.splitext(img_file)[0].replace(simdir, measdir)
                        cat_file = '%s%s'%(base,catext)
                        if run_check: check_file = '%s%s'%(base,checkext)
                        else: check_file=None
                        if skipdone:
                                if ( os.path.isfile(cat_file)):
                                        logger.info("%s exist sextractor run checking it"%(cat_file))
                                        
                                        try:
                                                if strongcheck:
                                                        if check_file is not None:
                                                                cat_fitsio=fitsio.read(check_file)
                                                        cat_fitsio=fitsio.read(cat_file)
                                                        logger.info("%s It is OK!!"%(cat_file))
                                                        continue
                                                else:
                                                        if check_file is not None:
                                                                assert (os.path.isfile(check_file))
                                                                assert bool(os.path.getsize(check_file))
                                                        assert (os.path.isfile(cat_file))
                                                        assert bool(os.path.getsize(cat_file))
                                                        logger.info("%s It is OK!!"%(cat_file))
                                                        continue
                                        except:
                                                logger.info("Something went wrong with image %s"%(check_file))
                                                logger.info("Trying measuring again")
                                                '''
                                                if check_file is not None:
                                                        if os.path.isfile(check_file):
                                                                os.remove(check_file)
                                                if os.path.isfile(cat_file): os.remove(cat_file)
                                                '''
                             

                                                '''
                                                if (not os.path.getsize(cat_file)):
                                                        os.remove(cat_file)
                                                        if check_file is not None:
                                                                if os.path.isfile(check_file):
                                                                        os.remove(check_file)
                                                else:
                                                        continue
                                                '''
                                

                        
                
                        ws = _SexWorkerSettings(img_file, cat_file, check_file, sex_bin, sex_config, sex_params, sex_filter, sex_nnw)
                        wslist.append(ws)
        
        logger.info("Ready to run Sextractor measurements on %i images." % (len(wslist)))
        _run(_sexworker, wslist, ncpu)
   
class _SexWorkerSettings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, img_file, cat_file, check_file, sex_bin, sex_config, sex_params, sex_filter,sex_nnw):
                
                self.img_file= img_file
                self.cat_file = cat_file
                self.check_file = check_file
                self.sex_bin = sex_bin
                self.sex_config = sex_config
                self.sex_params = sex_params
                self.sex_filter = sex_filter
                self.sex_nnw = sex_nnw

def _sexworker(ws):
        """
        Worker function that the different processes will execute, processing the
        _SexWorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        p = multiprocessing.current_process()
        logger.info("%s is starting Sextractor measure catalog %s with PID %s" % (p.name, str(ws), p.pid))
        
        sex.run_SEGMAP(ws.img_file, ws.cat_file, ws.check_file, ws.sex_bin,
                       ws.sex_config, ws.sex_params, ws.sex_filter,ws.sex_nnw)

        
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))

	
###########################
# Clean bad measurements  #
###########################
def checkandredosex(simdir, sex_bin, sex_config, sex_params, sex_filter, measdir, ext='_galimg.fits', catext="_cat.fits", checkext="_seg.fits"):
        import warnings
        warnings.filterwarnings("error")
        import astropy.io.fits
        import fitsio

        images_folders = sorted(glob.glob(os.path.join(simdir,'*_img') ))
        logger.info('The work dir have %i images'%(len(images_folders)))

        wslist = []

        
        for img_path in images_folders:
                filenames = sorted(glob.glob(os.path.join(img_path,'*%s'%(ext)) ))
                if (len(filenames) == 0):
                        logger.info("Skipping dir %s not %s files"%(img_path,ext))
                        continue

                #making measurmen dirs
                name = os.path.basename(img_path).replace("_img", "")
                imgdir_meas=os.path.join(measdir,"%s_img"%(name))
                        
                for img_file in filenames:
                        base = os.path.splitext(img_file)[0].replace(simdir, measdir)
                        cat_file = '%s%s'%(base,catext)
                        check_file = '%s%s'%(base,checkext)

                        try:
                                cat_fitsio=fitsio.read(cat)
                                cat_astropy=astropy.io.fits.open(cat)
                                cat_fitsio=fitsio.read(check_file)
                                cat_astropy=astropy.io.fits.open(check_file)
                        except:
                                logger.info("Something went wrong with image %s"%(check_file))
                                logger.info("Trying measuring again")
                                ws = _SexWorkerSettings(img_file, cat_file, check_file, sex_bin, sex_config, sex_params, sex_filter)
                                wslist.append(ws)
                                sex.run_SEGMAP(img_file, cat_file, check_file, sex_bin, sex_config, sex_params, sex_filter, logger)

        logger.info("Check and redo finished")
        
   


                
                
                        
