import warnings
warnings.filterwarnings("ignore")

import sys, os,  glob
import SHE_SIMS
from SHE_SIMS.utils import makedir, readyaml, group_measurements, open_fits, _run, write_to_file, read_integers_from_file, parallel_map
import argparse
import pandas as pd
import fitsio
import astropy.io.fits as fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)
import galsim
import datetime
import copy
import photutils
from photutils.aperture import CircularAperture, EllipticalAperture, aperture_photometry, ApertureStats
from photutils.isophote import EllipseGeometry, Ellipse
from astropy.modeling import models, fitting
import scipy

RAY=False
if RAY:
    import ray
    import ray.util.multiprocessing as multiprocessing
else:
    import multiprocessing
    import queue


def parse_args(): 
    parser = argparse.ArgumentParser(description='Basic code to produce simulations with neighbors')
    parser.add_argument('--workdir',
                        default='sim', 
                        help='diractory of work')

    # SIMS ARGs
    parser.add_argument('--constants',
                        default='/users/aanavarroa/original_gitrepos/euclid_sims/example/configfiles/simconstants.yaml',
                        help='yaml containing contant values in sims like ')
    parser.add_argument('--tru_type', default=2, type=int, 
                        help='Type of galaxy of model for the sims 0 gaussian, 1 sersic, 2bulge and disk')
    
    parser.add_argument('--ncat', default=10, type=int, 
                        help='Number of catalog to draw')
    parser.add_argument('--usevarpsf', default=False,
                        action='store_const', const=True, help='use variable psf')
    parser.add_argument('--usevarsky', default=False,
                        action='store_const', const=True, help='use variable Skybackground')
    parser.add_argument('--dist_type',
                        default='uniflagship', type=str,
                        help='type of distribution to draw simulation catalog')
    parser.add_argument('--selected_flagshipcat',
                        default=None,
                        help='Flagship catalog after the selection')
    parser.add_argument('--flagshippath',
                        default='/vol/euclidraid4/data/aanavarroa/catalogs/flagship/11542.fits',
                        #default='/vol/euclidraid5/data/aanavarroa/catalogs/flagship/widefield.fits',
                        help='diractory of work')
    parser.add_argument('--skycat',
                        default='/vol/euclidraid4/data/aanavarroa/catalogs/SC8/SkyLevel.dat', 
                        help='directory of work')
    parser.add_argument('--pixel_conv',
                        default=False, 
                        action='store_const', const=True,
                        help='If true then uses method auto in drawing. i.e the pixel response is applied')
    parser.add_argument('--max_shear',
                        default=0.05, type=float, 
                        help='max shear value in sims')
    parser.add_argument('--patience',
                        default=25, type=int, 
                        help='max number of attemps to measure a pair')
    parser.add_argument('--npairs',
                        default=2, type=int, 
                        help='Number of pairs to draw for each case')
    
    # ADAMOM ARGs
    parser.add_argument('--groupcats',
                        default=None,
                        help='Final output summing up all cats information')
    parser.add_argument('--subsample_nbins',
                        default=1, type=int, 
                        help='subsampling the pixel to avoid failure for sources being too small')
    # SETUPS ARGs
    
    parser.add_argument('--adamompsfcatalog',
                        default="/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits",
                        help='diractory of work')
    parser.add_argument('--drawcat', default=False,
                        action='store_const', const=True, help='Global flag for running only sims')
    parser.add_argument('--runadamom', default=False,
                        action='store_const', const=True, help='Global flag for running only adamom')
    parser.add_argument('--ncpu', default=2, type=int, 
                        help='Number of cpus')
    parser.add_argument('--skipdone', default=False,
                        action='store_const', const=True, help='Skip finished measures')
    parser.add_argument('--loglevel', default='INFO', type=str, 
                        help='log level for printing')



    
    args = parser.parse_args()
    return args



def calculate_concentration_index(image_data):
    # Calculate the total flux
    image_data = np.clip(image_data, a_min=0, a_max=None)
    
    total_flux = np.sum(image_data)

    # Calculate the center of mass of the image
    y_center, x_center = scipy.ndimage.center_of_mass(image_data)

    # Create a meshgrid for the image
    y, x = np.indices(image_data.shape)

    # Calculate the radial distance from the center
    radii = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)

    # Calculate the radial profile
    radial_profile, bin_edges = np.histogram(radii.ravel(), bins=50, weights=image_data.ravel())
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate cumulative flux
    cumulative_flux = np.cumsum(radial_profile)
    
    # Find r_20 and r_80
    r_20 = bin_centers[np.searchsorted(cumulative_flux, 0.2 * total_flux)]
    r_80 = bin_centers[np.searchsorted(cumulative_flux, 0.8 * total_flux)]
    
    return r_20, r_80


def drawinputcat(ncases, path=None, psfsourcecat=None, galscat=None, constants=None,  usevarpsf=True, usevarsky=True, skycat=None, tru_type=2, dist_type="uniflagship", max_shear=0.05 ):

    filename=os.path.join(path, "casescat.fits")
    if os.path.isfile(filename):
        logger.info("catalog alredy exist skiping it.")
        return
    
    psfdata=open_fits(psfsourcecat)
    psfdataconst=open_fits(psfsourcecat, hdu=2)
    constants["psf_path"]=psfdataconst["psf_path"][0]
    if usevarpsf:
        psfinfo= psfdata.sample(ncases)
    else:
        psfinfo= psfdata.loc[0]
        
    psfnames=list(psfdata.columns)
    psfformats=['U200' if e.type==np.object_  else e.type for e in psfdata.dtypes]

    if usevarsky:
        logger.info("Using variable skybackground")
        df=pd.read_csv(skycat)
        sky_vals=df["STRAY"]+df["ZODI"] #e/565s
        sky_vals=sky_vals.to_numpy()
    else:
        sky_vals=None
        
    if sky_vals is not None:
        sky=np.random.choice(sky_vals,ncases)/565.0 #the catalog with skyback is in electros for 565 s exposure
        tru_sky_level=(sky*constants["exptime"])/constants["realgain"]
    else:
        tru_sky_level= [(constants["skyback"]*constants["exptime"])/constants["realgain"]]*ncases

    gal= SHE_SIMS.sim.params_eu_gen.draw_sample(ncases, tru_type=tru_type,  dist_type=dist_type, sourcecat=galscat, constants=constants)
    tru_s1= np.random.uniform(-max_shear, max_shear, ncases)
    tru_s2= np.random.uniform(-max_shear, max_shear, ncases)

    gal.pop("tru_theta")
    gal.pop("tru_g1")
    gal.pop("tru_g2")
 
    names_const =  [ 'tru_type'] +list(constants.keys())
    values_const=[ tru_type] +[constants[k] for k in constants.keys()]
    formats_const = [ 'i4'] +[type(constants[k]) for k in constants.keys()]
    formats_const=[e if (e!=str)&(e!=np.str_) else 'U200' for e in formats_const]
    
    dtype_const = dict(names = names_const, formats=formats_const)
    outdata_const = np.recarray( (1, ), dtype=dtype_const)
    for i, key in enumerate(names_const):
        outdata_const[key] = values_const[i]

    names_var =  ['cat_id'] +["tru_s1","tru_s2"]+["sky_level"]+psfnames+ list(gal.keys())
    formats_var= ['i4']+['f4']*2 +['f4']+psfformats + [type(gal[g][0]) for g in gal.keys()]
    dtype_var = dict(names = names_var,
                     formats=formats_var)
    outdata_var = np.recarray((ncases, ), dtype=dtype_var)

    for key in gal.keys(): outdata_var[key] = gal[key]
    for key in psfnames: outdata_var[key] = psfinfo[key]
    
    outdata_var["cat_id"]=list(range(ncases))
    outdata_var["tru_s1"]=tru_s1
    outdata_var["tru_s2"]=tru_s2
    outdata_var["sky_level"]=tru_sky_level
    
    filename=os.path.join(path, "casescat.fits")
    hdul=[fits.PrimaryHDU(header=fits.Header())]
    hdul.insert(1, fits.BinTableHDU(outdata_var))
    hdul.insert(2, fits.BinTableHDU(outdata_const))
    hdulist = fits.HDUList(hdul)
    hdulist.writeto(filename, overwrite=True)
                

#random orientation pairs
def measure_pairs2(catid, row, constants, profile_type=2, npairs=1, subsample_nbins=1, filename=None, patience=1, blacklist=None):
    '''
    patience: max number of consecutive failures
    '''
    logger.info("\n")
    logger.info("Doing %s"%(filename))
    vispixelscale=0.1 #arcsec
    psfpixelscale = 0.02 #arcsec
    gsparams = galsim.GSParams(maximum_fft_size=100000)
    rng = galsim.BaseDeviate()
    ud = galsim.UniformDeviate()

    psffilename=os.path.join(constants["psf_path"][0],row["psf_file"])
    assert os.path.isfile(psffilename)
    with fits.open(psffilename) as hdul:
        psfarray=hdul[1].data
    psfarray_shape=psfarray.shape
    psfimg=galsim.Image(psfarray)            
    psf = galsim.InterpolatedImage( psfimg, flux=1.0, scale=psfpixelscale, gsparams=gsparams )


    profiles=["Gaussian", "Sersic", "EBulgeDisk"]
    profile_type=profiles[constants["tru_type"][0]]

    data=[]
    counter=0
    while (len(data)< npairs*2)&(counter<patience):
        tru_theta=np.random.uniform(0,1)*180.
        dataux=[]
        for rot in [False,True]:
            if profile_type == "Sersic":
                tru_rad=float(row["tru_rad"])
                tru_flux=float(row["tru_flux"])
                tru_sersicn=float(row["tru_sersicn"])
                
                if sersiccut is None:
                    trunc = 0
                else:
                    trunc = float(tru_rad) * sersiccut
                    gal = galsim.Sersic(n=tru_sersicn, half_light_radius=tru_rad, flux=tru_flux, gsparams=gsparams, trunc=trunc)
                # We make this profile elliptical
                gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"]) # This adds the ellipticity to the galaxy

            elif profile_type == "Gaussian":
                tru_sigma=float(row["tru_sigma"])
                tru_flux=float(row["tru_flux"])
                
                gal = galsim.Gaussian(flux=tru_flux, sigma=tru_sigma, gsparams=gsparams)
                # We make this profile elliptical
                gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"]) # This adds the ellipticity to the galaxy
        
            elif profile_type == "EBulgeDisk":
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
                #tru_theta=float(row["tru_theta"])

                bulge = galsim.Sersic(n=tru_bulge_sersicn, half_light_radius=tru_bulge_rad, flux=tru_bulge_flux, gsparams = gsparams)
                
                if False:
                    bulge_ell = galsim.Shear(g=tru_bulge_g, beta=tru_theta * galsim.degrees)
                    bulge = bulge.shear(bulge_ell)
                    gal=bulge
                        
                    if dominant_shape==1:
                        disk = galsim.InclinedExponential(inclination=tru_disk_inclination*galsim.degrees, 
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
            else:
                raise RuntimeError("Unknown galaxy profile!")

            theta=90. + tru_theta
            gal = gal.rotate(theta * galsim.degrees)
            if rot:
                gal = gal.rotate(90. * galsim.degrees)
                theta+=90.
            '''
            try:
                galtest=gal.withFlux(1.0)
                finestamp=galtest.drawImage(scale=psfpixelscale)
            except:
                counter+=1
                raise RuntimeError("Could not draw the input galaxy")
                break
                
            try: # First we try defaults:
                inputres = galsim.hsm.FindAdaptiveMom(finestamp, weight=None)
            except: # We change a bit the settings:
                try:
                    logger.debug("HSM defaults failed, retrying with larger sigma...")
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    inputres = galsim.hsm.FindAdaptiveMom(finestamp, guess_sig=15.0, hsmparams=hsmparams, weight=None)
                except:
                    counter+=1
                    raise RuntimeError("Could not measure the input galaxy")
                    break               
            tru_ada_sigma=inputres.moments_sigma
            '''
            
            '''
            try:
                tru_ada_sigma=gal.withFlux(1.0).calculateHLR()
            except:
                print(gal)
                couter+=patience
                break
                raise RuntimeError("Could not get the HLR from input galaxy")
            '''
            
            tru_ada_sigma=-1
        
            gal = gal.lens(float(row["tru_s1"]), float(row["tru_s2"]), 1.0)
            xjitter = vispixelscale*(ud() - 0.5)
            yjitter = vispixelscale*(ud() - 0.5)
            gal = gal.shift(xjitter,yjitter)
            galconv = galsim.Convolve([gal,psf])
            stamp = galconv.drawImage( scale=vispixelscale)

            stamp+=float(row["sky_level"])
            stamp.addNoise(galsim.CCDNoise(rng, sky_level=0.0,
                                           gain=float(constants["realgain"][0]),
                                           read_noise=float(constants["ron"][0])))

            edgewidth=5
            sky = SHE_SIMS.meas.utils.skystats(stamp, edgewidth)
            stamp-=sky["med"]

            if subsample_nbins>1:
                stamp=stamp.subsample(subsample_nbins,subsample_nbins)
            
            try:
                try: # First we try defaults:
                    res = galsim.hsm.FindAdaptiveMom(stamp, weight=None, guess_sigma=5.0*subsample_nbins)
                except: # We change a bit the settings:
                    logger.debug("HSM defaults failed, retrying with larger sigma...")
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    res = galsim.hsm.FindAdaptiveMom(stamp, guess_sig=15.0*subsample_nbins, hsmparams=hsmparams, weight=None)                        
            
            except: # If this also fails, we give up:
                logger.debug("Even the retry failed on:\n", exc_info = True)
                counter+=1
                break
    
            ada_flux = res.moments_amp
            ada_x = res.moments_centroid.x
            ada_y = res.moments_centroid.y 
            ada_g1 = res.observed_shape.g1
            ada_g2 = res.observed_shape.g2
            ada_sigma = res.moments_sigma
            ada_rho4 = res.moments_rho4
            centroid_shift=np.hypot(ada_x-stamp.true_center.x, ada_y-stamp.true_center.y)
            if centroid_shift>4:
                counter+=1
                break
                #continue
            skymad=sky["mad"]
            aper=3
            snr=(ada_flux*constants["realgain"][0])/(np.sqrt(ada_flux*constants["realgain"][0] + (np.pi*(ada_sigma*aper*1.1774*(1./subsample_nbins))**2) * (skymad*constants["realgain"][0])**2))
            features=[ada_flux, ada_sigma, ada_rho4, ada_g1,ada_g2, ada_x, ada_y,centroid_shift, skymad,snr, tru_ada_sigma, theta]
            #print(features, row["tru_mag"], row["bulge_r50"])
            dataux.append(features)
            if len(dataux)==2:
                data.append(dataux[0])
                data.append(dataux[1])
                counter=0

    if (counter>=patience):
        logger.info("Patience exceeded skiping this catalog")
        write_to_file(str(catid), blacklist)
        return
 
    names =  [ "adamom_%s"%(n) for n in  [ "flux", "sigma","rho4", "g1", "g2","x", "y"] ]+["centroid_shift", "skymad", "snr", "tru_adamom_sigma", "tru_theta"]
    formats =['f4']*len(names)
    dtype = dict(names = names, formats=formats)
    outdata = np.recarray( (len(data), ), dtype=dtype)
    for i, key in enumerate(names):
        outdata[key] = (np.array(data).T)[i]                 

    if filename is not None:
        fitsio.write(filename, outdata, clobber=True)
        logger.info("measurements catalog %s written"%(filename))
                              

def measure_pairs(catid, row, constants, profile_type=2, npairs=1, subsample_nbins=1, filename=None, patience=1, blacklist=None):
    '''
    patience: number of attempts to measure a pair
    '''
    logger.info("\n")
    logger.info("Doing %s"%(filename))
    vispixelscale=0.1 #arcsec
    psfpixelscale = 0.02 #arcsec
    gsparams = galsim.GSParams(maximum_fft_size=100000)
    rng = galsim.BaseDeviate()
    ud = galsim.UniformDeviate()

    psffilename=os.path.join(constants["psf_path"][0],row["psf_file"])
    assert os.path.isfile(psffilename)
    with fits.open(psffilename) as hdul:
        psfarray=hdul[1].data
    psfarray_shape=psfarray.shape
    psfimg=galsim.Image(psfarray)            
    psf = galsim.InterpolatedImage( psfimg, flux=1.0, scale=psfpixelscale, gsparams=gsparams )


    profiles=["Gaussian", "Sersic", "EBulgeDisk"]
    profile_type=profiles[constants["tru_type"][0]]

    data=[]
    counter=0
    delta_theta=90./npairs
    thetas_list=[i*delta_theta for i in range(npairs)]
    while (len(data)< npairs*2)&(counter<patience):
        tru_theta=thetas_list[0]
        dataux=[]
        for rot in [False,True]:
            if profile_type == "Sersic":
                tru_rad=float(row["tru_rad"])
                tru_flux=float(row["tru_flux"])
                tru_sersicn=float(row["tru_sersicn"])
                
                if sersiccut is None:
                    trunc = 0
                else:
                    trunc = float(tru_rad) * sersiccut
                    gal = galsim.Sersic(n=tru_sersicn, half_light_radius=tru_rad, flux=tru_flux, gsparams=gsparams, trunc=trunc)
                # We make this profile elliptical
                gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"]) # This adds the ellipticity to the galaxy

            elif profile_type == "Gaussian":
                tru_sigma=float(row["tru_sigma"])
                tru_flux=float(row["tru_flux"])
                
                gal = galsim.Gaussian(flux=tru_flux, sigma=tru_sigma, gsparams=gsparams)
                # We make this profile elliptical
                gal = gal.shear(g1=row["tru_g1"], g2=row["tru_g2"]) # This adds the ellipticity to the galaxy
        
            elif profile_type == "EBulgeDisk":
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
                #tru_theta=float(row["tru_theta"])

                bulge = galsim.Sersic(n=tru_bulge_sersicn, half_light_radius=tru_bulge_rad, flux=tru_bulge_flux, gsparams = gsparams)
                
                if False:
                    bulge_ell = galsim.Shear(g=tru_bulge_g, beta=tru_theta * galsim.degrees)
                    bulge = bulge.shear(bulge_ell)
                    gal=bulge
                        
                    if dominant_shape==1:
                        disk = galsim.InclinedExponential(inclination=tru_disk_inclination*galsim.degrees, 
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
            else:
                raise RuntimeError("Unknown galaxy profile!")

            theta=tru_theta
            gal = gal.rotate(theta * galsim.degrees)
            if rot:
                gal = gal.rotate(90. * galsim.degrees)
                theta+=90.
            '''
            try:
                galtest=gal.withFlux(1.0)
                finestamp=galtest.drawImage(scale=psfpixelscale)
            except:
                counter+=1
                raise RuntimeError("Could not draw the input galaxy")
                break
                
            try: # First we try defaults:
                inputres = galsim.hsm.FindAdaptiveMom(finestamp, weight=None)
            except: # We change a bit the settings:
                try:
                    logger.debug("HSM defaults failed, retrying with larger sigma...")
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    inputres = galsim.hsm.FindAdaptiveMom(finestamp, guess_sig=15.0, hsmparams=hsmparams, weight=None)
                except:
                    counter+=1
                    raise RuntimeError("Could not measure the input galaxy")
                    break               
            tru_ada_sigma=inputres.moments_sigma
            '''
            
            
            try:
                tru_ada_sigma=gal.withFlux(1.0).calculateHLR()
            except:
                print(gal)
                couter+=patience
                break
                raise RuntimeError("Could not get the HLR from input galaxy")
            
            
            
            tru_ada_sigma=-1

            stampsize=64
        
            gal = gal.lens(float(row["tru_s1"]), float(row["tru_s2"]), 1.0)
            xjitter = vispixelscale*(ud() - 0.5)
            yjitter = vispixelscale*(ud() - 0.5)
            gal = gal.shift(xjitter,yjitter)
            galconv = galsim.Convolve([gal,psf])
            
            stamp = galconv.drawImage( scale=vispixelscale, nx=stampsize, ny=stampsize)

            stamp+=float(row["sky_level"])
            stamp.addNoise(galsim.CCDNoise(rng, sky_level=0.0,
                                           gain=float(constants["realgain"][0]),
                                           read_noise=float(constants["ron"][0])))
                        
        
            edgewidth=5
            sky = SHE_SIMS.meas.utils.skystats(stamp, edgewidth)
            stamp-=sky["med"]

            if subsample_nbins>1:
                stamp=stamp.subsample(subsample_nbins,subsample_nbins)
            
            try:
                try: # First we try defaults:
                    res = galsim.hsm.FindAdaptiveMom(stamp, weight=None, guess_sigma=5.0*subsample_nbins)
                except: # We change a bit the settings:
                    logger.debug("HSM defaults failed, retrying with larger sigma...")
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    res = galsim.hsm.FindAdaptiveMom(stamp, guess_sig=15.0*subsample_nbins, hsmparams=hsmparams, weight=None)                        
            
            except: # If this also fails, we give up:
                logger.debug("Even the retry failed on:\n", exc_info = True)
                counter+=1
                break
    
            ada_flux = res.moments_amp
            ada_x = res.moments_centroid.x
            ada_y = res.moments_centroid.y 
            ada_g1 = res.observed_shape.g1
            ada_g2 = res.observed_shape.g2
            ada_sigma = res.moments_sigma
            ada_rho4 = res.moments_rho4
            centroid_shift=np.hypot(ada_x-stamp.true_center.x, ada_y-stamp.true_center.y)

            #print(res)
            positions=[(ada_x,ada_y)]
            radii = [5,10,15,20,25,30]
      
            error=photutils.utils.calc_total_error(stamp.array, sky["std"], float(constants["realgain"][0]))
            aper_data=[]
            aper_names=[]
            for r in radii:
                aperstats = ApertureStats(stamp.array, CircularAperture(positions, r=r*subsample_nbins), error=error)
                flux=aperstats.sum[0]
                flux_err=aperstats.sum_err[0]
                mad_std=aperstats.mad_std[0]
                cxx=aperstats.cxx
                cxy=aperstats.cxy
                cyy=aperstats.cyy
                elongation=aperstats.elongation
                gini=aperstats.gini[0]
                fwhm=aperstats.fwhm[0].value
                max=aperstats.max[0]
                std=aperstats.std[0]
                pars=[flux, flux_err, mad_std, cxx[0].value, cxy[0].value, cyy[0].value, elongation[0].value, gini, fwhm, max, std]
                aper_data+=pars
                aper_names+=['%s_r%i'%(st,r) for st in ['flux', 'flux_err', 'mad', 'cxx', 'cxy', 'cyy', 'elongation', 'gini','fwhm', 'max', 'std']]

                
            '''
            rad20, rad80= calculate_concentration_index(stamp.array)
            aper_data+= [rad20, rad80]
            aper_names+=['rad20', 'rad80']

            cutout_data = stamp.array
            print(cutout_data)
            #cutout_data = np.nan_to_num(cutout_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Create a meshgrid for the Sersic model
            y, x = np.indices(cutout_data.shape)

            # Initial guess for Sersic parameters
            initial_amplitude = np.max(cutout_data)
            initial_r_eff = ada_sigma  # Half-light radius in pixels
            initial_n = 4  # Initial guess for Sersic index
            initial_x0 = cutout_data.shape[0] / 2
            initial_y0 = cutout_data.shape[1]  / 2
            initial_ellip = np.hypot(ada_g1,ada_g2)  # Assuming circular for simplicity
            theta=np.arctan2(ada_g2, ada_g1)
            
            # Create the Sersic model
            sersic_model = models.Sersic2D(amplitude=initial_amplitude, r_eff=initial_r_eff, n=initial_n,
                                           x_0=initial_x0, y_0=initial_y0, ellip=initial_ellip,theta=theta)
            sersic_model.amplitude.bounds = (0, None)
            sersic_model.r_eff.bounds = (0, None)
            sersic_model.n.bounds = (0, 10)  # Example bounds for Sersic index
            sersic_model.ellip.bounds = (0, 1)  # Ellipticity must be between 0 and 1
            sersic_model.theta.bounds = (-np.pi, np.pi)  # Theta must be within -? to ?
            print(sersic_model)

            # Fit the Sersic model to the cutout data
            #fitter = fitting.LevMarLSQFitter()
            fitter = fitting.TRFLSQFitter()
            #fitter = fitting.LMLSQFitter()
            fitted_sersic = fitter(sersic_model, x, y, cutout_data, maxiter=1000, filter_non_finite=True)
            print(fitted_sersic)
            '''



            
            
            # Define the parameters for the elliptical aperture
            #a = ada_sigma  # Semi-major axis in pixels
            #b = ada_sigma    # Semi-minor axis in pixels
            #theta = 0.5*np.arctan2(ada_g2,ada_g1)  # Position angle in degrees
            # Create an elliptical aperture
            #aperture = EllipticalAperture(positions[0], a, b, theta=np.deg2rad(theta))
            #phot_table = aperture_photometry(stamp.array, aperture)
            #geometry = EllipseGeometry(x0=0.5*stamp.array.shape[0], y0=0.5*stamp.array.shape[1], sma=40, eps=0.5, pa=theta, astep=1.0)
            
            #ellipse = Ellipse(stamp.array, geometry)
            #ellipse.fit_image()

            '''
            cutout_data=stamp.array
            y, x = np.indices(cutout_data.shape)

            # Initial guess for the elliptical parameters
            initial_amplitude = np.max(cutout_data)
            initial_a = 10  # Semi-major axis in pixels
            initial_b = 5   # Semi-minor axis in pixels
            initial_x0 = cutout_data.shape[0] / 2
            initial_y0 = cutout_data.shape[1] / 2
            initial_theta = 0  # Position angle in radians
            
            # Create the Ellipse2D model
            ellipse_model = models.Ellipse2D(amplitude=initial_amplitude, x_0=initial_x0, 
                                             y_0=initial_y0, a=initial_a, b=initial_b, 
                                             theta=initial_theta)

            # Fit the elliptical model to the cutout data
            fitter = fitting.LevMarLSQFitter()
            fitted_ellipse = fitter(ellipse_model, x, y, cutout_data)
            print(fitted_ellipse)
            a,b=fitted_ellipse.a,fitted_ellipse.b
            print(a,b)
            print(np.hypot(ada_g1,ada_g2), (a-b)/(a+b))

            xx, yy = np.meshgrid(np.arange(cutout_data.shape[1]), np.arange(cutout_data.shape[0]))

            # Evaluate the fitted elliptical model
            fitted_data = fitted_ellipse(xx, yy)
            
            # Calculate the total flux by summing the fitted model values
            total_flux = np.sum(fitted_data)

            print(total_flux)
            '''

           
            #assert False


            
            if centroid_shift>4:
                counter+=1
                break
                #continue
            skymad=sky["mad"]
            aper=3
            snr=(ada_flux*constants["realgain"][0])/(np.sqrt(ada_flux*constants["realgain"][0] + (np.pi*(ada_sigma*aper*1.1774*(1./subsample_nbins))**2) * (skymad*constants["realgain"][0])**2))
            features=[ada_flux, ada_sigma, ada_rho4, ada_g1,ada_g2, ada_x, ada_y,centroid_shift, skymad,snr, tru_ada_sigma, theta]
            features+=aper_data
            #print(features, row["tru_mag"], row["bulge_r50"])
            dataux.append(features)
            if len(dataux)==2:
                data.append(dataux[0])
                data.append(dataux[1])
                counter=0
                thetas_list.pop(0)

    if (counter>=patience):
        logger.info("Patience exceeded skiping this catalog")
        write_to_file(str(catid), blacklist)
        return
 
    names =  [ "adamom_%s"%(n) for n in  [ "flux", "sigma","rho4", "g1", "g2","x", "y"] ]+["centroid_shift", "skymad", "snr", "tru_adamom_sigma", "tru_theta"]
    names += aper_names
    formats =['f4']*len(names)
    dtype = dict(names = names, formats=formats)
    outdata = np.recarray( (len(data), ), dtype=dtype)
    for i, key in enumerate(names):
        outdata[key] = (np.array(data).T)[i]                 

    if filename is not None:
        fitsio.write(filename, outdata, clobber=True)
        logger.info("measurements catalog %s written"%(filename))
  
def simmeas(inputcat, measdir=None, skipdone=True, ncpu=1, measkwargs=None, blacklist=None):
    cat=open_fits(inputcat,hdu=1)
    constants=open_fits(inputcat,hdu=2)
    blackids=read_integers_from_file(blacklist)
    
    wslist=[]
    for i, row in cat.iterrows():
        filename=os.path.join(measdir, "cat%i.fits"%(i))
        if i in blackids: continue
        if os.path.isfile(filename)&(skipdone):
            logger.info("File already exist skipping it")
            continue
        measkwargs["filename"]=filename
        #measure_pairs(row, constants, **measkwargs)
        ws = _WorkerSettings(i, row, constants, copy.deepcopy(measkwargs))
        wslist.append(ws)
    np.random.shuffle(wslist)
    if not RAY:
        _run(_worker, wslist,ncpu)
    else:
        parallel_map(_worker, wslist,ncpu)

class _WorkerSettings():
    def __init__(self, catid, row, constants, measkwargs):
        self.catid=catid
        self.row=row
        self.constants=constants
        self.measkwargs=measkwargs


if RAY:
    @ray.remote
    def _worker(ws):
        starttime = datetime.datetime.now()
        np.random.seed() #this is important
        
        measure_pairs(ws.catid, ws.row, ws.constants, **ws.measkwargs)
        endtime = datetime.datetime.now()
        print("cat %s is done, it took %s" % (ws.catid, str(endtime - starttime)))
else:
    def _worker(ws):
        starttime = datetime.datetime.now()
        np.random.seed() #this is important

        p = multiprocessing.current_process()
        logger.info("%s is starting measure catalog %s with PID %s" % (p.name, str(ws.catid),p.pid))

        measure_pairs(ws.catid, ws.row, ws.constants, **ws.measkwargs)
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))
  
        
def main():
    args = parse_args()

    if args.loglevel=='DEBUG':
        print("Using DEBUG logging")
        level = logging.DEBUG
    if args.loglevel=='INFO':
        print("Using INFO logging")
        level = logging.INFO
    if args.loglevel=='ERROR':
        print("Using ERROR logging")
        level = logging.ERROR
    if args.loglevel=='WARNING':
        print("Using WARNING logging")
        level = logging.WARNING
    

    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    logging.basicConfig(format=loggerformat, level=level)


    workdir = os.path.expanduser(args.workdir)
    makedir(workdir)
    


    #loading constants in sims like gain, exptime, ron, zeropoint,
    constants=readyaml(args.constants)
    assert os.path.isfile(args.adamompsfcatalog)
        
    #DRAW SIMULATION CATS AND IMAGES
    if (args.drawcat):
        logger.info("Drawing catalog")
        if (args.dist_type=='flagship')|(args.dist_type=='uniflagship'):
            original_sourceflagshipcat = args.flagshippath # to read
            if args.selected_flagshipcat is None:
                selected_flagshipcat = os.path.join(workdir, "selectedflagship.fits") #towrite
            else:
                selected_flagshipcat =args.selected_flagshipcat
                makedir(os.path.dirname(selected_flagshipcat))
                plotpath = os.path.join(os.path.dirname(selected_flagshipcat))
            
            if ~os.path.isfile(selected_flagshipcat):
                SHE_SIMS.sim.run_constimg_eu.drawsourceflagshipcat(original_sourceflagshipcat,selected_flagshipcat , workdir)

                
        caseargs={"path":workdir, 'psfsourcecat':args.adamompsfcatalog,
                  "galscat":selected_flagshipcat, 'constants':constants,
                  "usevarpsf":args.usevarpsf, "usevarsky":args.usevarsky, "skycat":args.skycat,
                  'tru_type':args.tru_type, "dist_type":args.dist_type, "max_shear": args.max_shear}
        drawinputcat(args.ncat, **caseargs )

    inputcat=os.path.join(workdir, "casescat.fits")
    filename=os.path.join(workdir, "groupcat.fits")
    blacklist=os.path.join(workdir, "blacklist.txt")
    if args.runadamom:        
        logger.info("Running adamom")
        measkwargs={"npairs":args.npairs,"patience":args.patience,"profile_type":args.tru_type, "subsample_nbins":args.subsample_nbins, "blacklist":blacklist}
        simmeas(inputcat, measdir=workdir, skipdone=args.skipdone, ncpu=args.ncpu, measkwargs=measkwargs, blacklist=blacklist)
    group_measurements(filename,inputcat,  workdir, picklecat=True)
            
   

if __name__ == "__main__":
    main()
    

