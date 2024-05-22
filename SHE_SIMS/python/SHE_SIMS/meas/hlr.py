"""
Add the true input half light radius. Specially important for multiple component profiles with inclinations
"""
import os, glob

import fitsio
import galsim
from astropy.io import fits
import numpy as np
from .. import utils

import datetime
RAY=False
if RAY:
        import ray
        import ray.util.multiprocessing as multiprocessing
else:
        import multiprocessing

import logging
logger = logging.getLogger(__name__)


#ADD NEIGHBOR FEATURES TO MEASURED CATALOG
def meas(catsdir,  ext='_cat.fits' , skipdone=False, ncpu=1):
        
        """
        catsdir: directory with the input catalogs
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        n_neis: include cols upto this number of neighbors (sorted by distance)
        
        """
        logger.info("Starting catalog measurement")

        input_catalogs=sorted(glob.glob(os.path.join(catsdir, '*%s'%(ext)) ))

        wslist=[]
        for catname in input_catalogs:
                logger.info("Doing cat %s"%(catname))
                wc=_Settings(catname)
                wslist.append(wc)
        utils._run(_worker, wslist, ncpu)
     
class _Settings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, cat_file):
                self.cat_file = cat_file
         
def _worker(ws):
        """
        Worker function that the different processes will execute, processing the _NEISSettings
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        p = multiprocessing.current_process()
        logger.info("%s is starting neibors measure catalog %s with PID %s" % (p.name, str(ws.cat_file), p.pid))

        catalog=utils.open_fits(ws.cat_file, hdu=1)
        const_cat=utils.open_fits(ws.cat_file, hdu=2)
        try:
                starcatalog=utils.open_fits(ws.cat_file, hdu=3)
        except:
                starcatalog=None
                
        profiles=["Gaussian", "Sersic", "EBulgeDisk"]
        vispixelscale=0.1 #arcsec
        psfpixelscale = 0.02 #arcsec
        gsparams = galsim.GSParams(maximum_fft_size=50000)
                
        tru_hlr_list=[]
        for row in catalog.to_records():
                profile_type=profiles[const_cat["tru_type"][0]]

                if profile_type == "Sersic":
                        if const_cat["snc_type"][0]==0:
                                tru_rad=float(row["tru_rad"])
                                tru_flux=float(row["tru_flux"])
                                tru_sersicn=float(row["tru_sersicn"])
                        else:
                                tru_rad=float(const_cat["tru_rad"][0])
                                tru_flux=float(const_cat["tru_flux"][0])
                                tru_sersicn=float(const_cat["tru_sersicn"][0])
                                
                        if sersiccut is None:
                                trunc = 0
                        else:
                                trunc = float(tru_rad) * sersiccut
                        gal = galsim.Sersic(n=tru_sersicn, half_light_radius=tru_rad, flux=tru_flux, gsparams=gsparams, trunc=trunc)
                        # We make this profile elliptical
                        gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"]) # This adds the ellipticity to the galaxy

                elif profile_type == "Gaussian":
                        if const_cat["snc_type"][0]==0:
                                tru_sigma=float(row["tru_sigma"])
                                tru_flux=float(row["tru_flux"])
                        else:
                                tru_sigma=float(const_cat["tru_sigma"][0])
                                tru_flux=float(const_cat["tru_flux"][0])
                                
                        gal = galsim.Gaussian(flux=tru_flux, sigma=tru_sigma, gsparams=gsparams)
                        # We make this profile elliptical
                        gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"]) # This adds the ellipticity to the galaxy
        
                elif profile_type == "EBulgeDisk":
                        if const_cat["snc_type"][0]==0:
                                bulge_axis_ratio=float(row['bulge_axis_ratio'])
                                tru_bulge_g=float(row["bulge_ellipticity"])
                                tru_bulge_flux=float(row["tru_bulge_flux"])
                                tru_bulge_sersicn=float(row["bulge_nsersic"])
                                tru_bulge_rad=float(row["bulge_r50"])
                                tru_disk_rad=float(row["disk_r50"])
                                tru_disk_flux=float(row["tru_disk_flux"])
                                tru_disk_inclination=float(row["inclination_angle"])
                                tru_disk_scaleheight=float(row["disk_scalelength"])
                                #tru_disk_scale_h_over_r=float(row["tru_disk_scale_h_over_r"])
                                dominant_shape=int(row["dominant_shape"])                         
                        else:
                                bulge_axis_ratio=float(const_cat['bulge_axis_ratio'][0])
                                tru_bulge_g=float(const_cat["bulge_ellipticity"][0])
                                tru_bulge_flux=float(const_cat["tru_bulge_flux"][0])
                                tru_bulge_sersicn=float(const_cat["bulge_nsersic"][0])
                                tru_bulge_rad=float(const_cat["bulge_r50"][0])
                                tru_disk_rad=float(const_cat["disk_r50"][0])
                                tru_disk_flux=float(const_cat["tru_disk_flux"][0])
                                tru_disk_inclination=float(const_cat["inclination_angle"][0])
                                tru_disk_scaleheight=float(const_cat["disk_scalelength"][0])
                                #tru_disk_scale_h_over_r=float(row["tru_disk_scale_h_over_r"][0])tru_rad=float(const_cat["tru_rad"][0][0])
                                dominant_shape=int(const_cat["dominant_shape"][0])
                        
                        # A more advanced Bulge + Disk model
                        # It needs GalSim version master, as of April 2017 (probably 1.5).

                        tru_theta=float(row["tru_theta"])
                        bulge = galsim.Sersic(n=tru_bulge_sersicn, half_light_radius=tru_bulge_rad, flux=tru_bulge_flux, gsparams = gsparams)

                        if False:
                                bulge_ell = galsim.Shear(g=tru_bulge_g, beta=tru_theta * galsim.degrees)
                                bulge = bulge.shear(bulge_ell)
                                gal=bulge
                        
                                if dominant_shape==1:
                                        disk = galsim.InclinedExponential(inclination=tru_disk_inclination* galsim.degrees , 
                                                                          half_light_radius=tru_disk_rad,
                                                                          flux=tru_disk_flux, 
                                                                          scale_height=tru_disk_scaleheight,
                                                                          gsparams = gsparams)
                                        disk = disk.rotate(tru_theta* galsim.degrees)                
                                        gal += disk
                        else:
                                bulge_ell = galsim.Shear(q=bulge_axis_ratio, beta=0 * galsim.degrees)
                                bulge = bulge.shear(bulge_ell)
                                gal=bulge
                                C_DISK=0.1
                        
                                if dominant_shape==1:
                                        incrad = np.radians(tru_disk_inclination)
                                        ba = np.sqrt(np.cos(incrad)**2 + (C_DISK * np.sin(incrad))**2)
                                        disk = galsim.InclinedExponential(inclination=tru_disk_inclination* galsim.degrees , 
                                                                          half_light_radius=tru_disk_rad/np.sqrt(ba),
                                                                          scale_h_over_r = C_DISK,
                                                                          flux=tru_disk_flux, 
                                                                          gsparams = gsparams)
                                        gal += disk
                                        
                                gal = gal.rotate((90. + tru_theta) * galsim.degrees)
                        
                                
                else:
                        raise RuntimeError("Unknown galaxy profile!")

                
                try:
                        print(gal)
                        tru_hlr=gal.calculateHLR()
                except:
                        tru_hlr=-1
                tru_hlr_list.append(tru_hlr)

        catalog["tru_hlr"]=tru_hlr_list
        #cat.loc[(cat['img_id']==img_id), 'ics']=ics

        tru_hlr_list=[]
        if starcatalog is not None:
                logger.debug("Including STARS in sims")
                psffilename=os.path.join(const_cat["psf_path"][0],const_cat["psf_file"][0])
                assert os.path.isfile(psffilename)
                with fits.open(psffilename) as hdul:
                        psfarray=hdul[1].data
                psfarray_shape=psfarray.shape
                psfimg=galsim.Image(psfarray)
                psf = galsim.InterpolatedImage( psfimg, flux=1.0, scale=psfpixelscale, gsparams=gsparams )

                for row in starcatalog.to_records():
                        try:
                                tru_hlr=psf.withFlux(row["tru_flux"]).calculateHLR()
                        except:
                                tru_hlr=-1
                        tru_hlr_list.append(tru_hlr)

                starcatalog["tru_hlr"]=tru_hlr_list
                

        
        catbintable = fits.BinTableHDU(catalog.to_records(index=False))
        starcatbintable = fits.BinTableHDU(starcatalog.to_records(index=False))
        with fits.open(ws.cat_file) as hdul:
                hdul.pop(1)
                hdul.insert(1, catbintable)
                hdul.pop(3)
                hdul.insert(3, starcatbintable)
                hdul.writeto(ws.cat_file, overwrite=True)
                
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))




