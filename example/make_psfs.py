import warnings
warnings.filterwarnings("ignore")

import sys, os,  glob
import galsim
import argparse
import astropy.io.fits as fits
import numpy as np
import time
from SHE_PSFToolkit.settings.share import initialise_settings
from SHE_PSFToolkit.settings import share, config
from SHE_PSFToolkit.model.state import PSFState
from SHE_PSFToolkit.weights.sed import SEDResponse_Pickles, SEDResponse_Galaxy
from SHE_PSFToolkit.toolkit import PSFGen

import logging
logger = logging.getLogger(__name__)

EXT="shiftUm2.0_big"

def parse_args(): 
    parser = argparse.ArgumentParser(description='Basic code to produce simulations with neighbors')
    parser.add_argument('--workdir',
                        default='/vol/euclidraid4/data/aanavarroa/catalogs/PSFs_PSFToolkit2022_%s/'%(EXT),
                        help='diractory of work')
    parser.add_argument('--configfile',
                        default=None, 
                        #default='/users/aanavarroa/original_gitrepos/SHE_PSFToolkit/SHE_PSFToolkit/auxdir/config/default_model.config', 
                        help='Config file for PSFToolkit')
    parser.add_argument('--redshiftcat',
                        #default=None,
                        default='/vol/euclidraid4/data/aanavarroa/catalogs/flagship/11542.fits', 
                        help='Catalog to extract the redshift distribution')
    parser.add_argument('--adamompsfcatalog',
                        default='/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_%s.fits'%(EXT), 
                        help='path where to save the measurements')
    parser.add_argument('--loglevel', default='INFO', type=str, 
                        help='level of logging')    
    args = parser.parse_args()
    return args


def make_psfs(configfile, image_save_path, redshiftcat=None,nsamples=2):
    tstart = time.time()
    temp_save_path = os.path.join(image_save_path, "tmp")

    cfg_overwrite = {"save_path":temp_save_path,
                     "pupil_telescope_geometry":"PDR",
                     "pupil_amplitude_antialias":False,
                     "image_coord":"-90T",
                     "pixelresponse_effect_switch":0,
                     "exposure_time":565,
                     #"telescope_model":"pdr00"
    }

    cfg_overwrite["save_opd_maps"] = "True"
    initialise_settings(configfile, cfg_overwrite)

    state = PSFState.nominal_factory(optical_mode="tmfit")
    state.get_optical_state().defocus(shiftUm = 2.0)   

    if redshiftcat is not None:
        zs=get_redshift_samples(redshiftcat, nsamples)
    else:
        zs=np.random.uniform(0.,3.0,size=nsamples)
        
    sed_names=np.random.choice(["el_cb2004a_001.fits", "sb2_b2004a_001.fits", "sb3_b2004a_001.fits", "sbc_cb2004a_001.fits", "scd_cb2004a_001.fits"], nsamples)
    seds = [SEDResponse_Galaxy.load_from_file("AUX/test_storage/galaxy_sed/%s"%(name), redshift = z) for name,z in zip(sed_names,zs)]

    # Basic - telescope model fit range (range of validity for distortion fit)
    fov_x_range = [-0.37350, 0.37350]
    fov_y_range = [0.52878, 1.15899]
    x=np.random.uniform(fov_x_range[0],fov_x_range[1],size=nsamples)
    y=np.random.uniform(fov_y_range[0],fov_y_range[1],size=nsamples)

    # From Euclidtelescopebaseline_iss1rev5, VIS FoV
    vis_fov_x_range = [-0.3935, 0.3935] #degrees
    vis_fov_y_range = [0.45, 1.159] #degrees
    vis_xmin=vis_fov_x_range[0] 
    vis_xmax=vis_fov_x_range[1]
    vis_ymin=vis_fov_y_range[0]
    vis_ymax=vis_fov_y_range[1]
    ccds=[get_virtual_ccd(x[i],y[i],vis_xmin,vis_xmax,vis_ymin,vis_ymax , n_ccd_x=6, n_ccd_y=6) for i in range(len(x))]
    quadrants=[get_virtual_ccd(x[i],y[i],vis_xmin,vis_xmax,vis_ymin,vis_ymax , n_ccd_x=2, n_ccd_y=2) for i in range(len(x))]

    generator = PSFGen(x, y, seds,state = state)

    tgenstart = time.time()
    options = dict(
        # Return oversampled images in the Fourier Domain? (FFT of asOS output)
        asFD = False,
        # If asFD, return with a mask showing where FD image is identically zero? 
        # (Can sometimes be used to speed up computation in post-processing)
        withFDMask = False,
        # Return detector sampled images?
        asDetector = False,
        # Return oversampled images?
        asOS = True,
        # Produce measures (ellipticity/size) for the images. If save_psf is used, this will output to the fits file
        # If get_psf is used, it will return the measures. Note that measures can be returned with save_psfs using
        # return_measures = True
        with_measures = True,
        # Print to screen a progress bar
        progress_bar = True
    )

    images, measures = generator.get_psf(**options)
    x_field=generator.x_field
    y_field=generator.y_field
    x_shift=generator.x_shift
    y_shift=generator.y_shift
    redshifts=[s.redshift for s in seds]

    priHDU = fits.PrimaryHDU()
    
    i=0
    for img, meas, x_f, y_f, x_s, y_s, ccd,q,z, seedname in zip(images['os'], measures['os'], x_field, y_field, x_shift, y_shift, ccds,quadrants, redshifts, sed_names):
        header={"X_FIELD":x_f, "Y_FIELD":y_f, "X_SHIFT":x_s, "Y_SHIFT":y_s, "id":meas["id"],
                "e1": meas["e1"], "e2":meas["e2"], "R2":meas["R2"], "ccd":ccd, "quadrant":q, "z":z }

        imHDU = fits.ImageHDU(img)
        add_header(imHDU, header)

        hdul = [priHDU, imHDU] 
        hdulist = fits.HDUList(hdul)

        filename=os.path.join(image_save_path, "seed%i_%s_ccd%i_z%.2f.os.fits"%(i,os.path.splitext(seedname)[0], ccd, z ))
        hdulist.writeto(filename, overwrite = True)
        logger.info("Image written to: %s \n "%(filename))
        hdulist.close()
        i+=1    
    
    tend = time.time()
    print("Time spent on model generation ", tend-tgenstart)
    print("Time from beginning to end: ", tend-tstart)

def make_adamom_psf_catalog(allpsffiles, filename):

    sed_names=["el_cb2004a_001", "sb2_b2004a_001", "sb3_b2004a_001", "sbc_cb2004a_001", "scd_cb2004a_001"]
    data=[]
    for f in allpsffiles:
        logger.info("Fitting %s"%(f))
        sed=[i for i,e in enumerate(sed_names) if e in f][0]

        with fits.open(f) as hdul:
            img_array=hdul[1].data
            hdul.close()
        img=galsim.Image(img_array, xmin=0, ymin=0)
        try: # We simply try defaults:
            print(img.array)
            res = galsim.hsm.FindAdaptiveMom(img, guess_centroid=None)
        except:
            logger.debug("HSM with default settings failed on", exc_info = True)
            adamom_list=[-999, -999, -999, -999, -999]
            data.append([f]+adamom_list)
            logger.info("Unable to measure %s"%(f))
            raise  # skip to next stamp !

        x_field=hdul[1].header["X_FIELD"]
        y_field=hdul[1].header["Y_FIELD"]
        ccd=hdul[1].header["ccd"]
        quadrant=hdul[1].header["quadrant"]
        e1=hdul[1].header["e1"]
        e2=hdul[1].header["e2"]
        R2=hdul[1].header["R2"]
        z=hdul[1].header["z"]
        toolkit_meas=[e1,e2,R2,z]
        
        ada_flux = res.moments_amp
        #ada_x = res.moments_centroid.x + 1.0 
        #ada_y = res.moments_centroid.y + 1.0
        ada_g1 = res.observed_shape.g1
        ada_g2 = res.observed_shape.g2
        ada_sigma = res.moments_sigma
        ada_rho4 = res.moments_rho4
            
        adamom_list=[ada_flux, ada_g1, ada_g2, ada_sigma, ada_rho4]
        data.append([f,quadrant, ccd, sed]+[x_field, y_field]+toolkit_meas+ adamom_list)

    #fits numpy ndarray 
    fields = ["flux", "g1", "g2", "sigma", "rho4"]
    names =  ["psf_file", "quadrant", "ccd", "sed"]+["X_FIELD", "Y_FIELD","e1","e2","R2","z"]+[ "%s%s"%("psf_adamom_", n) for n in fields ]
    formats = ['U200', 'i4', 'i4','i4' ]+['f4']*6 +['f4']*(len(fields))
    dtype = dict(names = names, formats=formats)
    outdata = np.recarray( (len(data), ), dtype=dtype)
    for i, key in enumerate(names):
        outdata[key] = (np.array(data).T)[i]
    
    if filename is not None:
        prihdu=fits.PrimaryHDU()
        table = fits.BinTableHDU( outdata)
        hdulist = fits.HDUList([prihdu,table])
        hdulist.writeto(filename, overwrite = True)
        logger.info("Image written to: %s \n "%(filename))
        hdulist.close()


def get_redshift_samples(catname, nsamples):
    with fits.open(catname) as hdul:
            cat=hdul[1].data

    #selection of galaxies
    mags=-2.5*np.log10(cat["euclid_vis"]) - 48.6
    mag_min=20
    mag_max=26
    magflag=(mags>mag_min)&(mags<mag_max)

    bulge_rad_min=0.1
    bulge_rad_max=1.0
    radflag=(cat["bulge_r50"]>bulge_rad_min)&(cat["bulge_r50"]< bulge_rad_max)

    cat=cat[magflag&radflag]
    
    zs=np.random.choice(cat["true_redshift_gal"], nsamples)
    return zs
    
def get_virtual_ccd(x,y, xmin, xmax, ymin, ymax, n_ccd_x=6, n_ccd_y=6):
    assert (x>xmin)&(x<xmax)
    assert (y>ymin)&(y<ymax)
    deltax=(xmax-xmin)/n_ccd_x
    deltay=(ymax-ymin)/n_ccd_y
    iccd_x=(x-xmin)//((xmax-xmin)/n_ccd_x)
    iccd_y=(y-ymin)//((ymax-ymin)/n_ccd_y)
    index = n_ccd_x*iccd_y + iccd_x + 1

    assert index%1==0
    
    plot=False
    if plot:
        plt.clf()
        plt.scatter(x,y,s=20, marker='o')
        (piy, pix) = np.vectorize(divmod)([ i for i in range(n_ccd_x*n_ccd_y)], n_ccd_x)
        x=xmin+pix*deltax
        y=ymin+piy*deltay
        
        plt.axhline(y=ymin, color='b', linestyle='-')
        plt.axhline(y=ymax, color='b', linestyle='-')
        plt.axvline(x=xmin, color='b', linestyle='-')
        plt.axvline(x=xmax, color='b', linestyle='-')
        for i in range(len(x)):
            plt.axhline(y=y[i], color='r', linestyle='-')
            plt.axvline(x=x[i], color='r', linestyle='-')

            
    return int(index)


def makedir(outpath):
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

def add_header(hdu, header):
    """
    Write header information to a particular HDU
    :param hdu:
    :param header: dictionary
    :return:
    """
    if header is None: return
    
    for key, val in header.items():
        hdu.header.set(key,val)


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
         
    make_psfs(args.configfile, workdir, args.redshiftcat, nsamples=10000)

    psf_files = sorted(glob.glob(os.path.join(workdir, "*.fits")))
    make_adamom_psf_catalog(psf_files, args.adamompsfcatalog)


        

  


if __name__ == "__main__":
    main()
    

