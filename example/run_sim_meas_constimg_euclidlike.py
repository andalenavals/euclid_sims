import warnings
warnings.filterwarnings("ignore")

import sys, os,  glob
import SHE_SIMS
import argparse
import pandas as pd
import fitsio
import astropy.io.fits as fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml 
import logging
logger = logging.getLogger(__name__)
import galsim


def parse_args(): 
    parser = argparse.ArgumentParser(description='Basic code to produce simulations with neighbors')
    parser.add_argument('--simdir',
                        default='sim', 
                        help='diractory of work')
    parser.add_argument('--sexdir',
                        default='sex', 
                        help='diractory of work')
    parser.add_argument('--adamomdir',
                        default='adamom', 
                        help='diractory of work')
    parser.add_argument('--ngmixdir',
                        default='ngmix', 
                        help='diractory of work')
    parser.add_argument('--ksbdir',
                        default='ksb', 
                        help='diractory of work')

    # SIMS ARGs
    parser.add_argument('--cat_args',
                        default='configfiles/tp-small.yaml',
                        help='yaml config define catalogs arguments')
    parser.add_argument('--constants',
                        default='configfiles/simconstants.yaml',
                        help='yaml containing contant values in sims like ')
    parser.add_argument('--tru_type', default=2, type=int, 
                        help='Type of galaxy of model for the sims 0 gaussian, 1 sersic, 2bulge and disk')
    parser.add_argument('--ncat', default=2, type=int, 
                        help='Number of catalog to draw')
    parser.add_argument('--usepsfimg', default=False,
                        action='store_const', const=True, help='use PSF img from Lance')
    parser.add_argument('--usevarpsf', default=False,
                        action='store_const', const=True, help='use variable psf')
    parser.add_argument('--usevarsky', default=False,
                        action='store_const', const=True, help='use variable Skybackground')
    parser.add_argument('--usevarshear', default=False,
                        action='store_const', const=True, help='use variable shear in image (for learning mu)')
    parser.add_argument('--dist_type',
                        default='uni', type=str,
                        help='type of distribution to draw simulation catalog')
    parser.add_argument('--selected_flagshipcat',
                        default=None,
                        help='Flagship catalog after the selection')
    parser.add_argument('--rot_pair', default=False,
                        action='store_const', const=True, help='Make rotated pair for validation')
    parser.add_argument('--transformtogrid', default=False,
                        action='store_const', const=True, help='Flag to make or not the catalogs that only transform catalogs')
    parser.add_argument('--transformtogriddir', default=None, help='Directory with catalogs that we want to put in the grid')
    parser.add_argument('--updatepsfsky', default=False,
                        action='store_const', const=True, help='Flag to use distribution in updatepsfskydir and update the psfs and skylevels')
    parser.add_argument('--updatepsfskydir', default=None, help='Directory with catalogs that we want to update psf and sky values')
    parser.add_argument('--flagshippath',
                        default='/vol/euclidraid4/data/aanavarroa/catalogs/flagship/11542.fits',
                        #default='/vol/euclidraid5/data/aanavarroa/catalogs/flagship/widefield.fits',
                        help='diractory of work')
    parser.add_argument('--psffilesdir',
                        default='/vol/euclid5/euclid5_raid3/mtewes/Euclid_PSFs_Lance_Jan_2020/', 
                        help='directory of work')
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
    
    
    # SEX ARGs
    parser.add_argument('--sex_args',
                        default='configfiles/sexconf.yaml',
                        #default='configfiles/sexconf_psfimg.yaml',
                        #default='configfiles/oldsexconf.yaml', 
                        help='yaml config define catalogs arguments')
    
    # ADAMOM ARGs
    parser.add_argument('--cattype',
                        #default='tru',
                        default='sex', type=str,
                        help='type of input catalog with initial position for galsim_adamom')
    parser.add_argument('--use_weight', default=False,
                        action='store_const', const=True, help='Use segmentation maps from sextractor as weights for galsim adamom')
    parser.add_argument('--typegroup', 
                        default='tp', 
                        help='Only for saving astropy pickle table for efficiency reasons')
    parser.add_argument('--nsubcases', default=0, type=int, 
                        help='Number of subcases')
    parser.add_argument('--groupcats',
                        default=None,
                        help='Final output summing up all cats information')
    parser.add_argument('--substractsky',
                        default=False, 
                        action='store_const', const=True,
                        help='Remove the median on hte edge from the stamp ')
    parser.add_argument('--match_pairs', default=False,
                        action='store_const', const=True, help='When saving the final catalog, only include completed measure pairs (exclude pairs if only one was sucessfully measured) requires measures to be on top of true positions, each position have an ID')

    # SETUPS ARGs
    
    parser.add_argument('--adamompsfcatalog',
                        default='/vol/euclid2/euclid2_raid2/aanavarroa/catalogs/MomentsML/fullimages_constimg_euclid/all_adamom_Lance_Jan_2020.fits', 
                        help='diractory of work')
    parser.add_argument('--runsims', default=False,
                        action='store_const', const=True, help='Global flag for running only sims')
    parser.add_argument('--runsex', default=False,
                        action='store_const', const=True, help='Global flag for running only sextractor')
    parser.add_argument('--runsexpp', default=False,
                        action='store_const', const=True, help='Global flag for running only sextractor++')
    parser.add_argument('--runngmix', default=False,
                        action='store_const', const=True, help='Running ngmix')
    parser.add_argument('--runadamom', default=False,
                        action='store_const', const=True, help='Global flag for running only adamom')
    parser.add_argument('--runksb', default=False,
                        action='store_const', const=True, help='Global flag for running only ksb')
    parser.add_argument('--runneis', default=False,
                        action='store_const', const=True, help='Add neighbor features to catalogs')
    parser.add_argument('--matchinput', default=False,
                        action='store_const', const=True, help='Add input features to meas cat matching')
    parser.add_argument('--stars', default=False,
                        action='store_const', const=True, help='Flag to measure the stars when using input catalog positions')
    parser.add_argument('--ncpu', default=1, type=int, 
                        help='Number of cpus')
    parser.add_argument('--nthreads', default=1, type=int, 
                        help='Number of threads used only by sourcextractor++')
    parser.add_argument('--skipdone', default=False,
                        action='store_const', const=True, help='Skip finished measures')
    parser.add_argument('--run_check', default=False,                                                                                                                                                            
                        action='store_const', const=True, help='run check image (i.e segmetantion map)')
    parser.add_argument('--add_ics', default=False,                                                                                                                                                            
                        action='store_const', const=True, help='Add input contamination statistic to the input catalog')
    parser.add_argument('--loglevel', default='INFO', type=str, 
                        help='log level for printing')



    
    args = parser.parse_args()
    return args



def group_measurements(filename, simdir, measdir, match_pairs, rot_pair, cols2d, cols1d, typegroup, nsubcases, constants=None, sexellip=True, picklecat=True, stars=False, cattype="tru"):
    #gal_density is include here while grouping
    if nsubcases>1:
        filename = filename.replace('.fits', '_subcase_ics2.fits')
        SHE_SIMS.group.cats_constimg(simdir, measdir, cols2d=cols2d,cols1d=cols1d, filename=filename, rot_pair=rot_pair, stars=stars, nsubcases=nsubcases, subcasecol='ICS2', cattype=cattype)
    else:
        if match_pairs&(cattype=="sex"):
            filename = filename.replace(".fits", "_matchpairs.fits")
            SHE_SIMS.group.cats_constimg_rotpair(simdir, measdir, cols2d=cols2d,cols1d=cols1d, stars=stars, filename=filename, cattype=cattype)
        else:
            if rot_pair: filename = filename.replace('.fits', '_rotpair.fits')
            SHE_SIMS.group.cats_constimg(simdir, measdir, cols2d=cols2d,cols1d=cols1d, filename=filename, rot_pair=rot_pair, stars=stars, cattype=cattype)

    #snr to the catalog
    SHE_SIMS.meas.snr.measfct(filename, gain=constants["realgain"])

    #SHE_SIMS.meas.snr.add_fluxsnr(filename) /it is the same that SNR_WIN
    #SHE_SIMS.meas.galsim_adamom.add_mag(filename)
   
    if sexellip:
        SHE_SIMS.meas.sex.add_ellipsepars(filename)
    
    
    if picklecat: SHE_SIMS.utils.makepicklecat(filename, typecat=typegroup, picklefilename=filename.replace(".fits",".pkl"),matchpairs=match_pairs)
    
    

def makedir(outpath):
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

def readyaml(filename):
    try:
        with open(filename) as file:
            aux= yaml.load(file, Loader=yaml.FullLoader)
            return aux
    except OSError :
            with open(filename) as file: raise
    

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


    simdir = os.path.expanduser(args.simdir)
    makedir(simdir)
    sexmeasdir = os.path.join(args.sexdir)
    adamomdir = os.path.join(args.adamomdir)
    ngmixdir = os.path.join(args.ngmixdir)
    ksbdir = os.path.join(args.ksbdir)

    #Loading catalogs arguments (i.e nimgs, snc, imagesize, positioning method)
    drawcatkwargs=readyaml(args.cat_args)
    #Loading sextractor++ arguments (i.e binary, config, python_config, params, conv_filter, checks)
    sex_args=readyaml(args.sex_args)
    #loading constants in sims like gain, exptime, ron, zeropoint,
    constants=readyaml(args.constants)

    
    assert os.path.isfile(args.adamompsfcatalog)

    if args.usevarsky:
        logger.info("Using variable skybackground")
        df=pd.read_csv(args.skycat)
        sky_vals=df["STRAY"]+df["ZODI"] #e/565s
        sky_vals=sky_vals.to_numpy()
    else:
        sky_vals=None
    drawcatkwargs.update({"sky_vals":sky_vals})
        
    #DRAW SIMULATION CATS AND IMAGES
    if (args.runsims):
        extra_caseargs={'dist_type':args.dist_type, 'psfsourcecat':args.adamompsfcatalog, 'tru_type':args.tru_type, "max_shear": args.max_shear, 'constants':constants, 'usevarpsf':args.usevarpsf}
    
        if (args.dist_type=='flagship')|(args.dist_type=='uniflagship'):
            original_sourceflagshipcat = args.flagshippath # to read
            if args.selected_flagshipcat is None:
                selected_flagshipcat = os.path.join(simdir, "selectedflagship.fits") #towrite
                plotpath = os.path.join(simdir)
            else:
                selected_flagshipcat =args.selected_flagshipcat
                makedir(os.path.dirname(selected_flagshipcat))
                plotpath = os.path.join(os.path.dirname(selected_flagshipcat))
            extra_caseargs.update({'sourcecat':selected_flagshipcat} )
            if (~os.path.isfile(selected_flagshipcat))&(not args.transformtogrid)&(not args.updatepsfsky):
                SHE_SIMS.sim.run_constimg_eu.drawsourceflagshipcat(original_sourceflagshipcat,selected_flagshipcat , plotpath)

        drawcatkwargs.update(extra_caseargs)

        if args.transformtogrid:
            logger.info("Using existing catalogs in %s to transform position to a grid"%(args.transformtogriddir))
            SHE_SIMS.sim.run_constimg_eu.transformtogrid(args.transformtogriddir, simdir, drawcatkwargs)

        if args.updatepsfsky:
            logger.info("Using existing catalogs in %s and updating psf and sky"%(args.transformtogriddir))
            SHE_SIMS.sim.run_constimg_eu.update_psf_sky(args.updatepsfskydir,simdir, drawcatkwargs)

        #gsparams = galsim.GSParams(maximum_fft_size=100000)
        gsparams = None
        constantshear= not args.usevarshear
        drawimgkwargs={"psfimg":args.usepsfimg, "rot_pair":args.rot_pair, "pixel_conv":args.pixel_conv, "gsparams":gsparams, "constantshear":constantshear}

        #print("used drawcatkwargs")
        #print(drawcatkwargs)
        print("used drawimgkwargs")
        print(drawimgkwargs)
        SHE_SIMS.sim.run_constimg_eu.multi(simdir, drawcatkwargs,drawimgkwargs, ncat=args.ncat , ncpu=args.ncpu,  skipdone=True,rot_pair=args.rot_pair, strongcheck=True)

        if args.runneis:
            neicols=["tru_flux","tru_rad", "tru_mag"]
            SHE_SIMS.meas.neighbors.measfct_trucat(simdir,ext='_cat.fits', cols=neicols,  n_neis=2, xname ='x', yname='y', r_label='tru_r', hdu=1, skipdone=args.skipdone, ncpu=args.ncpu)
        

    #if args.add_ics: SHE_SIMS.meas.neighbors.add_trueics(simdir)
    
    # RUNNING SEXTRACTOR
    if (args.runsex)|(args.runsexpp):
        logger.info("Running sextractor")
        sex_args.update({"skipdone": args.skipdone})
        #sex_args.update({"skipdone": False})
        if (args.runsex)& (~args.runsexpp):
            SHE_SIMS.meas.run.sextractor(simdir, **sex_args, measdir=sexmeasdir, ncpu=1, run_check=args.run_check, strongcheck=True)
        elif (~args.runsex)& (args.runsexpp):
            sex_args.update({"run_check":args.run_check})
            SHE_SIMS.meas.run.sextractorpp(simdir, **sex_args, measdir=sexmeasdir, ncpu=1, nthreads=args.nthreads )
        else:
            logger.error("You wanted to run two diferent version of sextractor")
            raise
        if args.runneis:
            neicols=["SNR_WIN", "MAG_AUTO"]
            SHE_SIMS.meas.neighbors.measfct(sexmeasdir,ext='_cat.fits', cols=neicols,  n_neis=2, xname ='X_IMAGE', yname='Y_IMAGE', r_label='SEX_R', hdu=2, skipdone=args.skipdone, ncpu=args.ncpu)

    if args.add_ics:
        SHE_SIMS.meas.neighbors.add_ics(sexmeasdir,simdir, ncpu=args.ncpu,skipdone=args.skipdone)
        if args.rot_pair:
            SHE_SIMS.meas.neighbors.add_ics(sexmeasdir,simdir, ext='_rot_cat.fits',ncpu=args.ncpu,skipdone=args.skipdone)
            
        
    cols1d, cols2d, extracols=SHE_SIMS.utils.get_cols1d_cols2d_extracols(args.typegroup, args.cattype)     
        
    if args.runadamom:        
        logger.info("Running adamom")
        if args.usevarshear:
            extracols+=["tru_s1", "tru_s2"]
            cols2d+=["tru_s1", "tru_s2"]
        measkwargs={"variant":"wider","skipdone":args.skipdone, "extra_cols":extracols, "use_weight":args.use_weight, "substractsky":args.substractsky, "edgewidth":5}
        if args.cattype=="tru": measkwargs.update({"stars":args.stars})
        print(measkwargs)

        #SHE_SIMS.meas.run.adamom(simdir,adamomdir, measkwargs, sexdir=sexmeasdir, cattype=args.cattype, ncpu=args.ncpu,  skipdone=args.skipdone, rot_pair=args.rot_pair)
        if args.runneis:
            neicols=["adamom_sigma", "adamom_flux"]
            SHE_SIMS.meas.neighbors.measfct(adamomdir,ext='_meascat.fits', cols=neicols,  n_neis=2,xname ='adamom_x', yname='adamom_y', r_label='adamom_r', skipdone=args.skipdone, ncpu=args.ncpu )

        if (args.matchinput):
            if args.typegroup == "tp":
                truecols=["x", "y", "tru_g1", "tru_g2"]
            if args.typegroup == "tw":
                truecols=["tru_flux","tru_g", "tru_mag", "obj_id", "x", "y", "tru_g1", "tru_g2"]+["tru_bulge_flux","tru_disk_flux", "tru_bulge_rad","tru_disk_rad", "tru_disk_inclination", "tru_bulge_sersicn", "tru_disk_scaleheight", "dominant_shape"]+["tru_r_n1", "tru_mag_n1"]
            #SHE_SIMS.meas.match.measfct(adamomdir,simdir,ext='_meascat.fits', cols=truecols, xname ='adamom_x', yname='adamom_y' )
            SHE_SIMS.meas.match.measfct(adamomdir,simdir,ext='_galimg_meascat.fits', cols=truecols, xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="", ncpu=args.ncpu, skipdone=args.skipdone )
            if args.rot_pair: SHE_SIMS.meas.match.measfct(adamomdir,simdir,ext='_galimg_rot_meascat.fits', cols=truecols, xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="", ncpu=args.ncpu, skipdone=args.skipdone,rot_pair=True )
        if (args.stars)&(args.cattype=="sex"):
            matchkwargs={ "xname":'X_IMAGE', "yname":'Y_IMAGE', "ncpu":args.ncpu, "skipdone":args.skipdone, "threshold":10.0, }
            SHE_SIMS.meas.match.stars(adamomdir,simdir,ext='_galimg_meascat.fits',  **matchkwargs )
            if args.rot_pair: SHE_SIMS.meas.match.stars(adamomdir,simdir,ext='_galimg_rot_meascat.fits', **matchkwargs)
 
        filename=os.path.join(adamomdir, args.groupcats)
        if args.cattype =='tru': sexellip=False
        elif args.cattype =='sex': sexellip=True

        picklecat=True
        if args.usevarshear: picklecat=False
        group_measurements(filename, simdir, adamomdir, args.match_pairs, args.rot_pair, cols2d, cols1d, args.typegroup, args.nsubcases, constants=constants, sexellip=sexellip, picklecat=picklecat, stars=args.stars, cattype=args.cattype)

        if args.usevarshear:
            cols1d.remove("tru_s1")
            cols1d.remove("tru_s2")
            SHE_SIMS.tools.utils.picklecat(filename, picklefilename=filename.replace(".fits",".pkl"), cols1d=cols1d, cols2d=cols2d)

    if args.runngmix:        
        logger.info("Running ngmix")
        # do not forget to use a conda env
        #measkwargs={"skipdone":args.skipdone, "extra_cols":extracols, "use_weight":args.use_weight, "substractsky":args.substractsky, "edgewidth":5, "usepsf":True, "method":"fit"}
        measkwargs={"skipdone":args.skipdone, "extra_cols":extracols, "use_weight":args.use_weight, "substractsky":args.substractsky, "edgewidth":5, "usepsf":True, "method":"moments"}
        print(measkwargs)
        SHE_SIMS.meas.run.ngmix_meas(simdir,ngmixdir, measkwargs, sexdir=sexmeasdir, cattype=args.cattype, ncpu=args.ncpu,  skipdone=args.skipdone, rot_pair=args.rot_pair)
        if args.runneis:
            neicols=["ngmix_T", "ngmix_flux"]
            SHE_SIMS.meas.neighbors.measfct(ngmixdir,ext='_meascat.fits', cols=neicols,  n_neis=2,xname ='adamom_x', yname='adamom_y', r_label='adamom_r', skipdone=args.skipdone, ncpu=args.ncpu )
        if (args.matchinput):
            #truecols=["tru_rad", "tru_sersicn", "tru_flux","tru_g", "tru_mag", "obj_id", "x", "y", "tru_g1", "tru_g2"]
            #truecols=["x", "y", "tru_g1", "tru_g2"]
            truecols=["tru_flux","tru_g", "tru_mag", "obj_id", "x", "y", "tru_g1", "tru_g2"]+["tru_bulge_flux","tru_disk_flux", "tru_bulge_rad","tru_disk_rad", "tru_disk_inclination", "tru_bulge_sersicn", "tru_disk_scaleheight", "dominant_shape"]
            #SHE_SIMS.meas.match.measfct(ksbdir,simdir,ext='_meascat.fits', cols=truecols, xname ='adamom_x', yname='adamom_y' )
            SHE_SIMS.meas.match.measfct(ngmixdir,simdir,ext='_galimg_meascat.fits', cols=truecols, xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="", ncpu=args.ncpu, skipdone=args.skipdone )
            SHE_SIMS.meas.match.measfct(ngmixdir,simdir,ext='_galimg_rot_meascat.fits', cols=truecols, xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="", ncpu=args.ncpu, skipdone=args.skipdone,rot_pair=True )

        filename=os.path.join(ngmixdir, args.groupcats)
        if args.cattype =='tru': sexellip=False
        elif args.cattype =='sex': sexellip=True
        picklecat=True
        if args.usevarshear: picklecat=False
        group_measurements(filename, simdir, ngmixdir, args.match_pairs, args.rot_pair, cols2d, cols1d, args.typegroup, args.nsubcases, constants=constants, sexellip=sexellip, picklecat=picklecat, stars=args.stars, cattype=args.cattype)

    if args.runksb:        
        logger.info("Running KSB")
        measkwargs={"variant":"wider", "skipdone":args.skipdone, "extra_cols":extracols, "use_weight":args.use_weight, "substractsky":args.substractsky, "edgewidth":5}
        SHE_SIMS.meas.run.ksb(simdir,ksbdir,  measkwargs,  sexdir=sexmeasdir, cattype=args.cattype, ncpu=args.ncpu,  rot_pair=args.rot_pair)
        if args.runneis:
            neicols=["ksb_T", "ksb_flux"]
            SHE_SIMS.meas.neighbors.measfct(ksbdir,ext='_meascat.fits', cols=neicols,  n_neis=2,xname ='adamom_x', yname='adamom_y', r_label='adamom_r', skipdone=args.skipdone, ncpu=args.ncpu )

        if (args.matchinput):
            truecols=["tru_flux","tru_g", "tru_mag", "obj_id", "x", "y", "tru_g1", "tru_g2"]+["tru_bulge_flux","tru_disk_flux", "tru_bulge_rad","tru_disk_rad", "tru_disk_inclination", "tru_bulge_sersicn", "tru_disk_scaleheight", "dominant_shape"]
            #SHE_SIMS.meas.match.measfct(ksbdir,simdir,ext='_meascat.fits', cols=truecols, xname ='adamom_x', yname='adamom_y' )
            SHE_SIMS.meas.match.measfct(ksbdir,simdir,ext='_galimg_meascat.fits', cols=truecols, xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="", ncpu=args.ncpu, skipdone=args.skipdone )
            SHE_SIMS.meas.match.measfct(ksbdir,simdir,ext='_galimg_rot_meascat.fits', cols=truecols, xname ='X_IMAGE', yname='Y_IMAGE', matchlabel="", ncpu=args.ncpu, skipdone=args.skipdone,rot_pair=True )


        filename=os.path.join(ksbdir, args.groupcats)
        if args.cattype =='tru': sexellip=False
        elif args.cattype =='sex': sexellip=True
        picklecat=True
        if args.usevarshear: picklecat=False
        group_measurements(filename, simdir, ksbdir, args.match_pairs, args.rot_pair, cols2d, cols1d, args.typegroup, args.nsubcases, constants=constants, sexellip=sexellip, picklecat=picklecat, stars=args.stars, cattype=args.cattype)  


if __name__ == "__main__":
    main()
    

