"""
Shape measurement with GalSim's adaptive moments (from its "hsm" module).
"""

import numpy as np
import sys, os
import fitsio
import pandas as pd
from astropy.io import fits
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import utils
from .. import calc
import galsim


profiles=["Gaussian", "Sersic", "EBulgeDisk"]
RLIST=[10,15,20,25,30]

def measure(imgfile, catalog,  xname="x", yname="y", variant="default", weight=None,  filename=None,  skystats=True, nmaskedpix=True,aperture_r=20, extra_cols=None, use_weight=True, rot_pair=False, skipdone=False, substractsky=True, edgewidth=5):
        """
        #catalog: catalog of objects in imgfile 
        #weight: weight image (segmentation map) to get neighbors features
        #use_weight: bool to use or not weight (when given) in the adaptive moments step
        #skytstats: flag to include skystats
        #nmaskedpix: flag to include segmap related features
        #aperture_r: radi around xname and yname to check the number of masked pixels
        #extra_cols: save cols from catalog 
        """
        starttime = datetime.now()
        
        if extra_cols is not None:
                extra_cols=[col for col in extra_cols if col in catalog.dtype.names]
                logger.debug("Adding %i extra cols in galsim_adamom measure"%(len(extra_cols)))
                if len(extra_cols)==0: extra_cols=None

        if skipdone:
                if os.path.isfile(filename):
                        try:
                                with fits.open(filename) as hdul:
                                        hdul.verify('fix')
                                        meascat=hdul[1].data
                                        #meascat=fitsio.read(filename,ext=2)
                                if extra_cols is not None:
                                        li=len(set(meascat.dtype.names).intersection(set(extra_cols)))
                                        if li==len(extra_cols):
                                                logger.info("Measurement file exist and have all existing extra column from sex cat")
                                                return
                                else:
                                        logger.info("Measurement file exist and there are not extra cols")
                                        return
                        except Exception as e:
                                logger.info("Unable to read existing measurement catalog %s, redoing it"%(filename))
                                logger.info(str(e))

        if len(catalog)==0:
                logger.info("Catalog %s have not galaxies. To noisy?"%(imgfile))
                return
        
        prefix="adamom_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                #print("Loading FITS image %s..." % (imgfile))
                logger.info("Loading FITS image %s..." % (imgfile))
                try:
                        img = galsim.fits.read(imgfile)
                        #img.setOrigin(0,0)
                        #galaxys were draw with the galsim and sextractor convention (1,1) 
                except:
                        logger.info("exception trying to read image %s"%(imgfile))
                        if os.path.exists(imgfile):
                                logger.info("removing corrupted image")
                                #os.remove(imgfile)
                                return None
                logger.debug("Done with loading %s, shape is %s" % (imgfile, img.array.shape))
                imagesize=img.array.shape[0]

        if (weight is not None) & (type(weight) is str):
                logger.debug("Loading FITS image weight %s..." % (os.path.basename(weight)))
                try:
                        with fits.open(weight) as hdul:
                                hdul.verify('fix')
                                data_seg=hdul[0].data
                        img_seg=galsim.Image(data_seg)
                except:
                        if os.path.exists(segmap_img):
                                logger.info("removing corrupted segmentation map")
                                #os.remove(segmap_img)
                                return None
                

       
        sigs_stamp=10
        stampsize=64
        # And loop
        data = []; data_sky=[]
        
        for i, gal in enumerate(catalog):

                flag=0
                        
                #logger.info("Using stampsize %i"%(stampsize))
                        
                (x, y) = (gal[xname], gal[yname]) # I set origin to (0,0)
                pos = galsim.PositionD(x,y)
                
                lowx=int(np.floor(x-0.5*stampsize))
                lowy=int(np.floor(y-0.5*stampsize))
                upperx=int(np.floor(x+0.5*stampsize))
                uppery=int(np.floor(y+0.5*stampsize))
                if lowx <1 :flag=1 ;lowx=1
                if lowy <1 :flag=1 ;lowy=1
                if upperx > imagesize : flag=1; upperx =imagesize
                if uppery > imagesize : flag=1; uppery =imagesize
                bounds = galsim.BoundsI(lowx,upperx , lowy , uppery ) # Default Galsim convention, index starts at 1.

                center = galsim.PositionI(int(np.floor(x)), int(np.floor(y)))
                if not bounds.includes(center):
                        logger.info("Galaxy out of bounds (%.2f,%.2f)"%(x,y))
                        continue
                
                if substractsky:
                        gps = img.copy()[bounds]
                else:
                        gps = img[bounds]
                sky = utils.skystats(gps, edgewidth)
                if substractsky:
                        gps-=sky["med"]
                        sky = utils.skystats(gps, edgewidth)

                if use_weight: assert (weight is not None)
                
                if weight is not None:
                        img_seg_stamp = img_seg[bounds]
                        
                        indx_use = [ 0, img_seg_stamp[center]]
                        mask = np.isin(img_seg_stamp.array,  indx_use) #True means use, False reject
                        gps_w = galsim.Image(mask*1, xmin=lowx,ymin=lowy)
                        if ~use_weight: gps_w=None

                        if nmaskedpix:
                                ny=uppery - lowy + 1
                                nx=upperx - lowx + 1
                                a, b = int(np.floor(y - lowy)), int(np.floor(x - lowx) )
                                fracpix=[]; fracpix_md=[]
                                for r in RLIST:
                                        #r=aperture_r
                                        yc,xc=np.ogrid[-a:ny-a, -b:nx-b]
                                        mask_c= xc*xc+yc*yc<=r*r
                                        tot_usedpix=np.sum(mask*mask_c )
                                        maskedpixels=np.sum(mask_c)-tot_usedpix
                                        fracpix.append(maskedpixels/tot_usedpix)

                                        X, Y = np.meshgrid(np.arange(nx) - b, np.arange(ny) - a)
                                        distance_ar = np.sqrt(X**2 + Y**2)
                                        Dinv = 1.0/np.clip( distance_ar, 1, aperture_r)

                                        fracpix_md.append(np.sum( (~mask)*(mask_c)*Dinv )/ np.sum(mask_c*Dinv))
                                        masked_D = np.clip(distance_ar, 1, aperture_r)*(~mask)
                                        if np.sum(masked_D) > 0:
                                                r_nmpix = np.min(masked_D[np.nonzero(masked_D)])
                                        else:
                                                logger.debug("Galaxy has not neighbors in the aperture of %i pixels"%(aperture_r))
                                                r_nmpix = r

                                galseg = np.isin(img_seg_stamp.array,  [img_seg_stamp[center]] )*1
                                galseg_area=np.sum(galseg)
                        
                else:
                        gps_w = None
                        nmaskedpix=False
                        assert ~use_weight
                        
                # And now we measure the moments... galsim may fail from time to time, hence the try:
                if variant == "default":
                        try: # We simply try defaults:
                                res = galsim.hsm.FindAdaptiveMom(gps,  weight=gps_w, guess_centroid=pos)
                        except:
                                # This is awesome, but clutters the output 
                                #logger.exception("GalSim failed on: %s" % (str(gal)))
                                # So instead of logging this as an exception, we use debug, but include the traceback :
                                logger.debug("HSM with default settings failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue # skip to next stamp !
                
                elif variant == "wider":

                        try:
                                try: # First we try defaults:
                                        res = galsim.hsm.FindAdaptiveMom(gps, weight=gps_w, guess_centroid=pos)
                                except: # We change a bit the settings:
                                        logger.debug("HSM defaults failed, retrying with larger sigma...")
                                        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                                        res = galsim.hsm.FindAdaptiveMom(gps, guess_sig=15.0, hsmparams=hsmparams, weight=gps_w,  guess_centroid=pos)                        

                        except: # If this also fails, we give up:
                                logger.debug("Even the retry failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue
                
                else:
                        raise RuntimeError("Unknown variant setting '{variant}'!".format(variant=variant))

                ada_flux = res.moments_amp
                ada_x = res.moments_centroid.x + 1.0 # Not fully clear why this +1 is needed. Maybe it's the setOrigin(0, 0).
                ada_y = res.moments_centroid.y + 1.0 # But I would expect that GalSim would internally keep track of these origin issues.
                ada_g1 = res.observed_shape.g1
                ada_g2 = res.observed_shape.g2
                ada_sigma = res.moments_sigma
                ada_rho4 = res.moments_rho4
                
                if ada_flux<0: continue
                

                adamom_list=[ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_sigma, ada_rho4]                        

                fields_names=[ "adamom_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma", "rho4"] ]
                if skystats:
                        sky_list=[sky["std"], sky["mad"], sky["mean"], sky["med"],sky["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, r_nmpix, galseg_area]+fracpix+ fracpix_md,
                        adamom_list+=neis_list
                        fields_names+=['mpix', 'r_nmpix', 'galseg_area']+['fracpix%i'%(i) for i in RLIST]+['fracpix_md%i'%(i) for i in RLIST]
                if extra_cols is not None:
                        if ("tru_g1" in extra_cols)&("tru_g2" in extra_cols)&(rot_pair):
                                logger.debug("rotating tru_g1 and tru_g2 before saving. Extracols are: %s"%("".join(extra_cols)))
                                sncrot=90
                                aux1,aux2=calc.rotg(gal["tru_g1"],gal["tru_g2"],sncrot)
                                gal["tru_g1"]=aux1
                                gal["tru_g2"]=aux2
                        extra_cols_list=[gal[n] for n in extra_cols]
                        adamom_list+=extra_cols_list
                        fields_names+=extra_cols
                        
                data.append([flag]+adamom_list)

        if (len(data)==0):
                logger.info("None of the detected galaxies was measured")
                return
        
        names =  ['flag']+fields_names
        formats =['i4']+[type(a) for a in adamom_list]
        dtype = dict(names = names, formats=formats)
        outdata = np.recarray( (len(data), ), dtype=dtype)
        for i, key in enumerate(names):
                outdata[key] = (np.array(data).T)[i]                 

        if filename is not None:
                fitsio.write(filename, outdata, clobber=True)
                logger.info("measurements catalog %s written"%(filename))


        endtime = datetime.now()        
        logger.info("All done")

        n = len(catalog)
        nfailed = n - len(data)
        
        logger.info("I failed on %i out of %i sources (%.1f percent)" % (nfailed, n, 100.0*float(nfailed)/float(n)))
        logger.info("This measurement took %.3f ms per galaxy" % (1e3*(endtime - starttime).total_seconds() / float(n)))


def measure_withsex(imgfile, catname, xname="X_IMAGE", yname="Y_IMAGE", variant="default", weight=None,  filename=None, skystats=True, nmaskedpix=True, aperture_r=20, extra_cols=None, use_weight=True, rot_pair=False, skipdone=False, substractsky=True, edgewidth=5):
        """
        #catalog: catalog of objects of the imgfie to measure
        #weight: weight image (segmentation map) to get neighbors features
        #use_weight: bool to use or not weight (when given) in the adaptive moments step
        #skytstats: flag to include skystats
        #nmaskedpix: flag to include number of masked pixels
        #aperture_r: radi around xname and yname to check the number of masked pixels
        #extra_cols: save cols from catalog 
        """
        starttime = datetime.now()
                
        try:
                with fits.open(catname) as hdul:
                        hdul.verify('fix')
                        catalog=hdul[2].data
                #catalog=fitsio.read(catname,ext=2)
        except Exception as e:
                logger.info("Skipping measurement. Unable to read catalog %s, removing it"%(catname))
                logger.info(str(e))
                if os.path.isfile(catname): os.remove(catname)
                return

        if len(catalog)==0:
                logger.info("WARNING sextractor catalog for %s have not galaxies. To noisy?"%(imgfile))
                return

        
        
        if extra_cols is not None:
                extra_cols=[col for col in extra_cols if col in catalog.dtype.names]
                logger.debug("Adding %i extra cols in galsim_adamom measure using sextractor catalog"%(len(extra_cols)))
                if len(extra_cols)==0: extra_cols=None
                        
        
        if weight is None: nmaskedpix=False
        
        prefix="adamom_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                try:
                        img = galsim.fits.read(imgfile)
                        logger.debug("Done with loading %s, shape is %s" % (imgfile, img.array.shape))
                except:
                        logger.info("exception trying to read image %s"%(imgfile))
                        if os.path.exists(imgfile):
                                logger.info("removing corrupted image")
                                os.remove(imgfile)
                                return None
                imagesize=img.array.shape[0]

        if (weight is not None) & (type(weight) is str):
                logger.debug("Loading FITS image %s..." % (os.path.basename(weight)))
                try:
                        with fits.open(weight) as hdul:
                                hdul.verify('fix')
                                data_seg=hdul[0].data
                        img_seg=galsim.Image(data_seg)
                except:
                        if os.path.exists(segmap_img):
                                logger.info("removing corrupted segmentation map")
                                os.remove(segmap_img)
                                return None

       
        sigs_stamp=10
        min_stampsize=64; max_stampsize=128

        # And loop
        data = []; data_sky=[]
        for j, gal in enumerate(catalog):
                stampsize=int(sigs_stamp*gal["FLUX_RADIUS"])

                if stampsize < min_stampsize:
                        stampsize=min_stampsize
                if stampsize > max_stampsize:
                        stampsize=max_stampsize

                flag=0
                        
                #logger.info("Using stampsize %i"%(stampsize))
                        
                (x, y) = (gal[xname], gal[yname]) #sextractor coordinates beginng in (1.0,1.0)
                pos = galsim.PositionD(x,y)
                
                lowx=int(np.floor(x-0.5*stampsize)) 
                lowy=int(np.floor(y-0.5*stampsize))
                upperx=int(np.floor(x+0.5*stampsize))
                uppery=int(np.floor(y+0.5*stampsize))
                if lowx <1 :flag=1 ;lowx=1
                if lowy <1 :flag=1 ;lowy=1
                if upperx > imagesize : flag=1; upperx =imagesize
                if uppery > imagesize : flag=1; uppery =imagesize
                bounds = galsim.BoundsI(lowx,upperx , lowy , uppery ) # Default Galsim convention, index starts at 1.

                center = galsim.PositionI(int(np.floor(x)), int(np.floor(y)))
                if not bounds.includes(center):
                        logger.info("Galaxy out of bounds (%.2f,%.2f)"%(x,y))
                        continue
                
                if substractsky:
                        gps = img.copy()[bounds]
                else:
                        gps = img[bounds]
                sky = utils.skystats(gps, edgewidth)
                if substractsky:
                        gps-=sky["med"]
                        sky = utils.skystats(gps, edgewidth)

                if use_weight: assert (weight is not None)
                if weight is not None:
                        img_seg_stamp = img_seg[bounds]
                        indx_use = [ 0, img_seg_stamp[center]]
                        mask = np.isin(img_seg_stamp.array,  indx_use)
                        gps_w = galsim.Image(mask*1, xmin=lowx,ymin=lowy)
                        if ~use_weight: gps_w=None

                        if nmaskedpix:
                                ny=uppery - lowy + 1
                                nx=upperx - lowx + 1
                                a, b = int(np.floor(y - lowy)), int(np.floor(x - lowx) )
                                fracpix=[]; fracpix_md=[]
                                for r in RLIST:
                                        #r=aperture_r
                                        yc,xc=np.ogrid[-a:ny-a, -b:nx-b]
                                        mask_c= xc*xc+yc*yc<=r*r
                                        tot_usedpix=np.sum(mask*mask_c )
                                        maskedpixels=np.sum(mask_c)-tot_usedpix
                                        fracpix.append(maskedpixels/tot_usedpix)

                                        X, Y = np.meshgrid(np.arange(nx) - b, np.arange(ny) - a)
                                        distance_ar = np.sqrt(X**2 + Y**2)
                                        Dinv = 1.0/np.clip( distance_ar, 1, aperture_r)

                                        fracpix_md.append(np.sum( (~mask)*(mask_c)*Dinv )/ np.sum(mask_c*Dinv))
                                        masked_D = np.clip(distance_ar, 1, aperture_r)*(~mask)
                                        if np.sum(masked_D) > 0:
                                                r_nmpix = np.min(masked_D[np.nonzero(masked_D)])
                                        else:
                                                logger.debug("Galaxy has not neighbors in the aperture of %i pixels"%(aperture_r))
                                                r_nmpix = r

                                galseg = np.isin(img_seg_stamp.array,  [img_seg_stamp[center]] )*1
                                galseg_area=np.sum(galseg)
                                #test_img = galsim.Image((~mask)*1, xmin=lowx,ymin=lowy)*(mask_c)*Dinv
                                #filename_test="%s_obj%i_distance.fits"%(os.path.basename(imgfile), j)
                                #test_img.write(filename_test)

                        
                        #Computationally expensive for large images
                        #seg_indx= np.array([0]+[ data_seg[int(gal[yname]),int(gal[xname])] ])
                        #a=[ (data_seg==seg_indx[i]) for i in range(len(seg_indx))]
                        #mask=np.where(np.bitwise_or.reduce(a), 1, 0)
                        #img_wgt=galsim.Image(mask, xmin=0,ymin=0)
                        #gps_w = img_wgt[bounds]
                        #if nmaskedpix:
                        #        a,b=int(y),int(x)
                        #        n=mask.shape[0]
                        #        r=aperture_r
                        #        yc,xc=np.ogrid[-a:n-a, -b:n-b]
                        #        mask_c= xc*xc+yc*yc<=r*r
                        #        tot_usedpix=np.sum(img_wgt.array*mask_c)
                        #        maskedpixels=np.sum(mask_c)-tot_usedpix
                        #        fracpix=maskedpixels/tot_usedpix
                        #        #print(tot_usedpix,maskedpixels)
                        
                                
                        # Some test plots for the masking
                        #filename_test="%s_obj%i_masked.fits"%(os.path.basename(imgfile), j)
                        #test_img=gps*gps_w*mask_c
                        #test_img=gps*gps_w
                        #test_img.write(filename_test)
                else:
                        gps_w = None
                        nmaskedpix=False
     
                # And now we measure the moments... galsim may fail from time to time, hence the try:
                if variant == "default":
                        try: # We simply try defaults:
                                res = galsim.hsm.FindAdaptiveMom(gps,  weight=gps_w, guess_centroid=pos)
                        except:
                                # This is awesome, but clutters the output 
                                #logger.exception("GalSim failed on: %s" % (str(gal)))
                                # So instead of logging this as an exception, we use debug, but include the traceback :
                                logger.debug("HSM with default settings failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue # skip to next stamp !
                
                elif variant == "wider":

                        try:
                                try: # First we try defaults:
                                        res = galsim.hsm.FindAdaptiveMom(gps, weight=gps_w, guess_centroid=pos)
                                except: # We change a bit the settings:
                                        logger.debug("HSM defaults failed, retrying with larger sigma...")
                                        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                                        res = galsim.hsm.FindAdaptiveMom(gps, guess_sig=15.0, hsmparams=hsmparams, weight=gps_w,  guess_centroid=pos)                        

                        except: # If this also fails, we give up:
                                logger.debug("Even the retry failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue
                
                else:
                        raise RuntimeError("Unknown variant setting '{variant}'!".format(variant=variant))

                gal_id = gal["NUMBER"]-1
                ada_flux = res.moments_amp
                ada_x = res.moments_centroid.x + 1.0 # Not fully clear why this +1 is needed. Maybe it's the setOrigin(0, 0).
                ada_y = res.moments_centroid.y + 1.0 # But I would expect that GalSim would internally keep track of these origin issues.
                ada_g1 = res.observed_shape.g1
                ada_g2 = res.observed_shape.g2
                ada_sigma = res.moments_sigma
                ada_rho4 = res.moments_rho4
                
                #saving only features if there were not failed measures
                if ada_flux<0: continue
                
                adamom_list=[ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_sigma, ada_rho4]                        

                fields_names=[ "adamom_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma", "rho4"] ]
                if skystats:
                        sky_list=[sky["std"], sky["mad"], sky["mean"], sky["med"],sky["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, r_nmpix, galseg_area]+fracpix+ fracpix_md
                        adamom_list+=neis_list
                        fields_names+=['mpix', 'r_nmpix', 'galseg_area']+['fracpix%i'%(i) for i in RLIST] + ['fracpix_md%i'%(i) for i in RLIST]
                if extra_cols is not None:
                        if ("tru_g1" in extra_cols)&("tru_g2" in extra_cols)&(rot_pair):
                                sncrot=90
                                aux1,aux2=calc.rotg(gal["tru_g1"],gal["tru_g2"],sncrot)
                                gal["tru_g1"]=aux1
                                gal["tru_g2"]=aux2
                        extra_cols_list=[gal[n] for n in extra_cols]
                        adamom_list+=extra_cols_list
                        fields_names+=extra_cols
                        
                data.append([gal_id, flag]+adamom_list)

        if (len(data)==0):
                logger.info("None of the detected galaxies was measured")
                return
        
        names =  ['obj_number', 'flag']+fields_names
        formats =['i4', 'i4']+[type(a) for a in adamom_list]
        dtype = dict(names = names, formats=formats)
        outdata = np.recarray( (len(data), ), dtype=dtype)
        for i, key in enumerate(names):
                outdata[key] = (np.array(data).T)[i]                 

        if filename is not None:
                fitsio.write(filename, outdata, clobber=True)
                logger.info("measurements catalog %s written"%(filename))


        endtime = datetime.now()        
        logger.info("All done")

        n = len(catalog)
        nfailed = n - len(data)
        
        logger.info("I failed on %i out of %i sources (%.1f percent)" % (nfailed, n, 100.0*float(nfailed)/float(n)))
        logger.info("This measurement took %.3f ms per galaxy" % (1e3*(endtime - starttime).total_seconds() / float(n)))

        
#injecting neighbors
def measure_withinjection(imgfile, catalog, ninjections=None , xname="x", yname="y",  variant="default", weight=None,  filename=None,  skystats=True, nmaskedpix=True,aperture_r=20, extra_cols=None, use_weight=True):
        """
        skytstats: flag to include skystats
        ninjections: number of injections per stamp
        """
        starttime = datetime.now()
        if extra_cols is not None:
                extra_cols=[col for col in extra_cols if col in catalog.dtype.names]
                logger.debug("Adding %i extra cols in galsim_adamom measure using sextractor catalog"%(len(extra_cols)))
                if len(extra_cols)==0: extra_cols=None
                
        prefix="adamom_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                img = galsim.fits.read(imgfile)
                img.setOrigin(0,0)
                logger.debug("Done with loading %s, shape is %s" % (imgfile, img.array.shape))
                imagesize=img.array.shape[0]

        if (weight is not None) & (type(weight) is str):
                logger.debug("Loading FITS image %s..." % (os.path.basename(weight)))
                segmap_img=weight
                data_seg=fitsio.read(segmap_img)
                img_seg=galsim.Image(data_seg, xmin=0,ymin=0)

       
        sigs_stamp=10
        stampsize=64
        # And loop
        data = []; data_sky=[]
        
        for i, gal in enumerate(catalog):
               

                flag=0
                        
                #logger.info("Using stampsize %i"%(stampsize))
                        
                (x, y) = (gal[xname], gal[yname])
                pos = galsim.PositionD(x,y)
                
                lowx=int(x-0.5*stampsize)+1
                lowy=int(y-0.5*stampsize)+1
                upperx=int(x+0.5*stampsize)
                uppery=int(y+0.5*stampsize)
                if lowx <0 :flag=1 ;lowx=0
                if lowy <0 :flag=1 ;lowy=0
                if upperx >= imagesize : flag=1; upperx =imagesize-1
                if uppery >= imagesize : flag=1; uppery =imagesize-1
                bounds = galsim.BoundsI(lowx,upperx , lowy , uppery ) # Default Galsim convention, index starts at 1.
                gps = img[bounds]
                if ninjections is not None:
                        vispixelscale=0.1 #arcsec
                        psfpixelscale = 0.02
                        psf = galsim.Gaussian(flux=1., sigma=3.55*psfpixelscale)
                        psf = psf.shear(g1=-0.0374, g2=9.414e-06 )
                        gals=np.random.choice(catalog,ninjections)
                        for row in gals:
                                gal = galsim.Sersic(n=row["tru_sersicn"], half_light_radius=row["tru_rad"], flux=row["tru_flux"], gsparams=gsparams, trunc=0)
                                gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"])
                                pos = galsim.PositionD(np.random.uniform(lowx,upperx), np.random.uniform(lowy,uppery))
                                galconv = galsim.Convolve([gal,psf])
                                galconv.drawImage(gps, add_to_image=True, center=pos,scale=vispixelscale)
                        
                if weight is not None:
                        img_seg_stamp = img_seg[bounds]
                        #center = galsim.PositionI(int(x), int(y))
                        center = galsim.PositionI(int(np.floor(x-1)), int(np.floor(y-1)))
                        indx_use = [ 0, img_seg_stamp[center]]
                        mask = np.isin(img_seg_stamp.array,  indx_use)*1
                        gps_w = galsim.Image(mask, xmin=lowx,ymin=lowy)
                        if ~use_weight: gps_w=None

                        if nmaskedpix:
                                ny=uppery - lowy + 1
                                nx=upperx - lowx + 1
                                a, b = int(np.floor(y - lowy)), int(np.floor(x - lowx) )
                                r=aperture_r
                                yc,xc=np.ogrid[-a:ny-a, -b:nx-b]
                                mask_c= xc*xc+yc*yc<=r*r
                                tot_usedpix=np.sum(mask*mask_c )
                                maskedpixels=np.sum(mask_c)-tot_usedpix
                                fracpix=maskedpixels/tot_usedpix

                                X, Y = np.meshgrid(np.arange(nx) - b, np.arange(ny) - a)
                                distance_ar = np.sqrt(X**2 + Y**2)
                                Dinv = 1.0/np.clip( distance_ar, 1, aperture_r)

                                fracpix_md = np.sum( (~mask)*(mask_c)*Dinv )/ np.sum(mask_c*Dinv)
                                masked_D = np.clip(distance_ar, 1, aperture_r)*(~mask)
                                if np.sum(masked_D) > 0:
                                        r_nmpix = np.min(masked_D[np.nonzero(masked_D)])
                                else:
                                        logger.debug("Galaxy has not neighbors in the aperture of %i pixels"%(aperture_r))
                                        r_nmpix = aperture_r

                                galseg = np.isin(img_seg_stamp.array,  [img_seg_stamp[center]] )*1
                                galseg_area=np.sum(galseg)
                        
                else:
                        gps_w = None
                        
                # And now we measure the moments... galsim may fail from time to time, hence the try:
                if variant == "default":
                        try: # We simply try defaults:
                                res = galsim.hsm.FindAdaptiveMom(gps,  weight=gps_w, guess_centroid=pos)
                        except:
                                # This is awesome, but clutters the output 
                                #logger.exception("GalSim failed on: %s" % (str(gal)))
                                # So instead of logging this as an exception, we use debug, but include the traceback :
                                logger.debug("HSM with default settings failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue # skip to next stamp !
                
                elif variant == "wider":

                        try:
                                try: # First we try defaults:
                                        res = galsim.hsm.FindAdaptiveMom(gps, weight=gps_w, guess_centroid=pos)
                                except: # We change a bit the settings:
                                        logger.debug("HSM defaults failed, retrying with larger sigma...")
                                        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                                        res = galsim.hsm.FindAdaptiveMom(gps, guess_sig=15.0, hsmparams=hsmparams, weight=gps_w,  guess_centroid=pos)                        

                        except: # If this also fails, we give up:
                                logger.debug("Even the retry failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue
                
                else:
                        raise RuntimeError("Unknown variant setting '{variant}'!".format(variant=variant))

                ada_flux = res.moments_amp
                ada_x = res.moments_centroid.x + 1.0 # Not fully clear why this +1 is needed. Maybe it's the setOrigin(0, 0).
                ada_y = res.moments_centroid.y + 1.0 # But I would expect that GalSim would internally keep track of these origin issues.
                ada_g1 = res.observed_shape.g1
                ada_g2 = res.observed_shape.g2
                ada_sigma = res.moments_sigma
                ada_rho4 = res.moments_rho4
                
                if ada_flux<0: continue
                

                adamom_list=[ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_sigma, ada_rho4]                        

                fields_names=[ "adamom_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma", "rho4"] ]
                if skystats:
                        out = utils.skystats(gps)
                        sky_list=[out["std"], out["mad"], out["mean"], out["med"],out["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, fracpix, fracpix_md, r_nmpix, galseg_area]
                        adamom_list+=neis_list
                        fields_names+=['mpix', 'fracpix', 'fracpix_md', 'r_nmpix', 'galseg_area']
                if extra_cols is not None:
                        extra_cols_list=[gal[n] for n in extra_cols]
                        adamom_list+=extra_cols_list
                        fields_names+=extra_cols
                        
                data.append([ flag]+adamom_list)

        if (len(data)==0):
                logger.info("None of the detected galaxies was measured")
                return
        
        names = ['flag']+fields_names
        formats = ['i4']+[type(a) for a in adamom_list]
        dtype = dict(names = names, formats=formats)
        outdata = np.recarray( (len(data), ), dtype=dtype)
        for i, key in enumerate(names):
                outdata[key] = (np.array(data).T)[i]                 

        if filename is not None:
                fitsio.write(filename, outdata, clobber=True)
                logger.info("measurements catalog %s written"%(filename))


        endtime = datetime.now()        
        logger.info("All done")

        n = len(catalog)
        nfailed = n - len(data)
        
        logger.info("I failed on %i out of %i sources (%.1f percent)" % (nfailed, n, 100.0*float(nfailed)/float(n)))
        logger.info("This measurement took %.3f ms per galaxy" % (1e3*(endtime - starttime).total_seconds() / float(n)))


def add_mag(catalog, zeropoint=24.6, gain=3.1, exptime=3.0*565.0):
        logger.info("Getting adamom_mag")
        #tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
        #tru_mag=zeropoint-2.5*np.log10(tru_flux*(gain/exptime))
        
        output = fitsio.read(catalog)
        output= output.astype(output.dtype.newbyteorder('='))
        output = pd.DataFrame(output)
        output["adamom_mag"]=zeropoint-2.5*np.log10(output["adamom_flux"]*(gain/exptime))
        
        fitsio.write(catalog, output.to_records(index=False), clobber=True)
