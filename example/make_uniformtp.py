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

RAY=True
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
                
def measure_pairs(catid, row, constants, profile_type=2, npairs=1, subsample_nbins=1, filename=None, patience=1, blacklist=None):
    '''
    patience: number of attempts to measure a pair
    '''
    logger.info("\n")
    logger.info("Doing %s"%(filename))
    vispixelscale=0.1 #arcsec
    psfpixelscale = 0.02 #arcsec
    gsparams = galsim.GSParams(maximum_fft_size=50000)
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

        
            gal = gal.rotate((90. + tru_theta) * galsim.degrees)
            if rot: gal = gal.rotate(90. * galsim.degrees)

            
            try:
                finestamp=gal.drawImage(scale=psfpixelscale)
            except:
                continue
                
            try: # First we try defaults:
                inputres = galsim.hsm.FindAdaptiveMom(finestamp, weight=None)
            except: # We change a bit the settings:
                try:
                    logger.debug("HSM defaults failed, retrying with larger sigma...")
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    inputres = galsim.hsm.FindAdaptiveMom(finestamp, guess_sig=15.0, hsmparams=hsmparams, weight=None)
                except:
                    continue                
            tru_ada_sigma=inputres.moments_sigma

        
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
                    res = galsim.hsm.FindAdaptiveMom(stamp, weight=None)
                except: # We change a bit the settings:
                    logger.debug("HSM defaults failed, retrying with larger sigma...")
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    res = galsim.hsm.FindAdaptiveMom(stamp, guess_sig=15.0, hsmparams=hsmparams, weight=None)                        
            
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
            if centroid_shift>2: continue
            skymad=sky["mad"]
            aper=3
            snr=(ada_flux*constants["realgain"][0])/(np.sqrt(ada_flux*constants["realgain"][0] + (np.pi*(ada_sigma*aper*1.1774)**2) * (skymad*constants["realgain"][0])**2))
            features=[ada_flux, ada_sigma, ada_rho4, ada_g1,ada_g2, ada_x, ada_y,centroid_shift, skymad,snr, tru_ada_sigma]
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
 
    names =  [ "adamom_%s"%(n) for n in  [ "flux", "sigma","rho4", "g1", "g2","x", "y"] ]+["centroid_shift", "skymad", "snr", "tru_adamom_sigma"]
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
    #_run(_worker, wslist,ncpu)
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
                  'tru_type':args.tru_type, "dist_type":"uniflagship", "max_shear": args.max_shear}
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
    

