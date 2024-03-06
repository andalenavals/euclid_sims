"""
Shape measurement with GalSim's adaptive moments (from its "hsm" module).
"""

import numpy as np
import sys, os
import fitsio
from astropy.io import fits
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import utils
from .. import calc
import galsim


profiles=["Gaussian", "Sersic", "EBulgeDisk"]
RLIST=[10,15,20,25,30]

def measure(imgfile, psffile,  catalog,  xname="x", yname="y", variant="default", weight=None,  filename=None,  skystats=True, nmaskedpix=True,aperture_r=20, extra_cols=None, use_weight=True, rot_pair=False, skipdone=False, substractsky=True, edgewidth=5):
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
                logger.debug("Adding %i extra cols in galsim_ksb measure using catalog columns"%(len(extra_cols)))
                if len(extra_cols)==0: extra_cols=None

        if skipdone:
                if os.path.isfile(filename):
                        try:
                                with fits.open(filename) as hdul:
                                        hdul.verify('fix')
                                        meascat=hdul[1].data
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
        
        prefix="ksb_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                #print("Loading FITS image %s..." % (imgfile))
                logger.info("Loading FITS image %s..." % (imgfile))
                try:
                        img = galsim.fits.read(imgfile)
                except:
                        logger.info("exception trying to read image %s"%(imgfile))
                        if os.path.exists(imgfile):
                                logger.info("removing corrupted image")
                                #os.remove(imgfile)
                                return None
                logger.debug("Done with loading %s, shape is %s" % (imgfile, img.array.shape))
                imagesize=img.array.shape[0]
        
        try:
                vispixelscale=0.1 #arcsec
                psfpixelscale = 0.02 #arcsec
                nbins=int(vispixelscale/psfpixelscale)
                with fits.open(psffile) as hdul:
                        psfarray=hdul[1].data
                psfimg=galsim.Image(psfarray, scale=psfpixelscale)#.bin(nbins,nbins)
                gsparams = galsim.GSParams(maximum_fft_size=50000)
                psf = galsim.InterpolatedImage( psfimg, flux=1.0, scale=psfpixelscale, gsparams=gsparams )
                pix=galsim.Pixel(vispixelscale)
                psf=galsim.Convolve(psf,pix)
                psfimg=psf.drawImage(method="no_pixel")
                
                #print(type(psf))
                #print(psf.array.shape)
                
                #psf.setOrigin(0,0)
        except Exception as e:
                logger.error("unable to read psffile %s"%(psffile))
                raise RuntimeError(str(e))

        #assert False

        if (weight is not None) & (type(weight) is str):
                logger.debug("Loading FITS image weight %s..." % (os.path.basename(weight)))
                segmap_img=weight
                try:
                        with fits.open(segmap_img) as hdul:
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
                        
                (x, y) = (gal[xname], gal[yname])
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
                        center = galsim.PositionI(int(x), int(y))

                        if not bounds.includes(center):
                                logger.info("Galaxy out of bounds")
                                continue
                        
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
                                                r_nmpix = aperture_r

                                galseg = np.isin(img_seg_stamp.array,  [img_seg_stamp[center]] )*1
                                galseg_area=np.sum(galseg)
                        
                else:
                        gps_w = None
                        nmaskedpix=False
                        assert ~use_weight
                        
                # And now we measure the moments... galsim may fail from time to time, hence the try:
                gps_sub=gps.subsample(nbins,nbins)
                if gps_w is not None:
                        gps_w=gps_w.subsample(nbins,nbins) #not need to convert to 0,1 that happens internally
                if flag==1:
                        pos=(pos-gps.origin)*nbins+galsim.PositionD(1.0,1.0)
                else:
                        pos=None
                        
                if variant == "default":
                        try: # We simply try defaults:
                                res = galsim.hsm.EstimateShear(gps_sub, psfimg, weight=gps_w, guess_centroid=pos, shear_est='KSB', sky_var=sky["std"]**2)
                        except:
                                # This is awesome, but clutters the output 
                                #logger.exception("GalSim failed on: %s" % (str(gal)))
                                # So instead of logging this as an exception, we use debug, but include the traceback :
                                logger.debug("HSM with default settings failed on:\n %s" % (str(gal)), exc_info = True)              
                                continue # skip to next stamp !
                
                elif variant == "wider":
                        try:
                                #psfimg and img do not have the same image size
                                try: # First we try defaults:
                                        res = galsim.hsm.EstimateShear(gps_sub, psfimg, weight=gps_w, guess_centroid=pos, shear_est='KSB', sky_var=sky["std"]**2)
                                        
                                except: # We change a bit the settings:
                                        logger.debug("HSM defaults failed, retrying with larger sigma...")
                                        hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                                        #res = galsim.hsm.FindAdaptiveMom(gps, guess_sig=15.0, hsmparams=hsmparams, weight=gps_w,  guess_centroid=pos) 
                                        res = galsim.hsm.EstimateShear(gps_sub, psfimg, guess_sig_gal=15.0, hsmparams=hsmparams, weight=gps_w,  guess_centroid=pos, shear_est='KSB', sky_var=sky["std"]**2)

                                logger.debug("Successful measurement was done")
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

                corr_s1=res.corrected_g1
                corr_s2=res.corrected_g2
                psf_sigma=res.psf_sigma
                psf_g1=res.psf_shape.g1
                psf_g2=res.psf_shape.g2
                
                if ada_flux<0: continue
                

                adamom_list=[ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_sigma, ada_rho4, corr_s1, corr_s2, psf_sigma, psf_g1, psf_g2]                        

                fields_names=[ "adamom_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma", "rho4"]] +["corr_s1", "corr_s2","psf_sigma_ksb", "psf_g1_ksb", "psf_g2_ksb" ]
                if skystats:
                        sky_list=[sky["std"], sky["mad"], sky["mean"], sky["med"],sky["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, r_nmpix, galseg_area] + fracpix + fracpix_md
                        adamom_list+=neis_list
                        fields_names+=['mpix', 'r_nmpix', 'galseg_area']+['fracpix%i'%(i) for i in RLIST]+['fracpix_md%i'%(i) for i in RLIST]
                if extra_cols is not None:
                        if ("tru_g1" in extra_cols)&("tru_g2" in extra_cols)&(rot_pair):
                                logger.info("rotating tru_g1 and tru_g2 before saving")
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
