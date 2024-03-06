"""
Shape measurement with Ngmix.
"""

import numpy as np
import sys, os
import fitsio
from astropy.io import fits
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import utils
import galsim


MAX_CENTROID_SHIFT = 1.0

BORDER_IMG= 1
BAD_MEASUREMENT = 2
CENTROID_SHIFT = 4
OUTLIER = 8
FAILURE = 32

RLIST=[10,15,20,25,30]

def make_ngmix_prior(T, pixel_scale,rng):
    import ngmix
    from ngmix import priors, joint_prior
    # centroid is 1 pixel gaussian in each direction
    cen_prior=priors.CenPrior(0.0, 0.0, pixel_scale, pixel_scale, rng=rng)

    # g is Bernstein & Armstrong prior with sigma = 0.1
    gprior=priors.GPriorBA(0.1, rng=rng)

    # T is log normal with width 0.2
    Tprior=priors.LogNormal(T, 0.2, rng=rng)

    # flux is the only uninformative prior
    Fprior=priors.FlatPrior(-10.0, 1.e10, rng=rng)

    prior=joint_prior.PriorSimpleSep(cen_prior, gprior, Tprior, Fprior)
    return prior


#Without psf step
def ngmix_fit(im, psf_im, wt, radius, x, y, usepsf=True, method="moments"):
    '''
    radius: method is moments correspond to the apperture, if not the the guess size of the galaxy
    '''
    import ngmix

    #logger.info('fwhm = %s, T_guess = %s'%(fwhm, T_guess))
    #wcs = im.wcs.local(im.center)
    #cen = im.true_center - im.origin
    #jac = ngmix.Jacobian(wcs=wcs, x=cen.x + x - int(x+0.5), y=cen.y + y - int(y+0.5))
    
    
    jac=None
    if ngmix.__version__ > '1.3.9':
        flag = 0
        g1,g2,g1_err,g2_err =0,0,0,0
        try:
            if usepsf:
                psf_obs = ngmix.Observation(psf_im.array, weight=None, jacobian=None )
                if wt is None:
                    obs = ngmix.Observation(image=im.array, jacobian=jac, psf=psf_obs)
                else:
                    obs = ngmix.Observation(image=im.array, weight=wt.array, jacobian=jac, psf=psf_obs)
                    
                boot=get_obs_bootstrapper(radius, usepsf=usepsf,method=method)
                res = boot.go(obs)
            else:
                if wt is None:
                    obs = ngmix.Observation(image=im.array, jacobian=jac)
                else:
                    obs = ngmix.Observation(image=im.array, weight=wt.array, jacobian=jac)  
                boot=get_obs_bootstrapper(radius, usepsf=usepsf,method=method)
                res = boot.go(obs)
            if (method=="moments"):
                output = {key: res[key] for key in ["flux","flux_err","T","T_err","s2n"]}
                g1,g2=res["e"]
                g1_err,g2_err=res["e_err"]
            if (method=="fit"):
                output = {key: res[key] for key in ["T","T_err","s2n"]}
                g1,g2=res["g"]
                g1_err,g2_err=res["g_err"]
        except Exception as e:
            logger.info(e)
            logger.info(' *** Bad measurement (caught exception).  Mask this one.')
            flag |= BAD_MEASUREMENT
            return flag, None,g1,g2,g1_err,g2_err 

        if res['flags'] != 0:
            logger.info(' *** Bad measurement (ngmix flag = %d).  Mask this one.',output['flags'])
            flag |= BAD_MEASUREMENT
        return flag,output,g1,g2,g1_err,g2_err  
            
            
    if ngmix.__version__ <= '1.3.9':
        wcs=galsim.PixelScale(0.1)
        flag = 0
        dx, dy, g1, g2, flux = 0., 0., 0., 0., 0.
        T_guess = (fwhm / 2.35482)**2 * 2.
        T = T_guess
        prior = make_ngmix_prior(T, wcs.minLinearScale())
        try:
            if wt is None:
                obs = ngmix.Observation(image=im.array, jacobian=jac)
            else:
                obs = ngmix.Observation(image=im.array, weight=wt.array, jacobian=jac)
            lm_pars = {'maxfev':4000}
            runner=ngmix.bootstrap.PSFRunner(obs, 'gauss', T, lm_pars, prior=prior)
            runner.go(ntry=3)
            ngmix_flag = runner.fitter.get_result()['flags']
            gmix = runner.fitter.get_gmix()
        except Exception as e:
            logger.info(e)
            logger.info(' *** Bad measurement (caught exception).  Mask this one.')
            flag |= BAD_MEASUREMENT
            return dx,dy,g1,g2,T,flux,flag
        
        if ngmix_flag != 0:
            logger.info(' *** Bad measurement (ngmix flag = %d).  Mask this one.',ngmix_flag)
            flag |= BAD_MEASUREMENT

        dx, dy = gmix.get_cen()
        if dx**2 + dy**2 > MAX_CENTROID_SHIFT**2:
            logger.info(' *** Centroid shifted by %f,%f in ngmix.  Mask this one.',dx,dy)
            flag |= CENTROID_SHIFT

        g1, g2, T = gmix.get_g1g2T()
        if abs(g1) > 0.5 or abs(g2) > 0.5:
            logger.info(' *** Bad shape measurement (%f,%f).  Mask this one.',g1,g2)
            flag |= BAD_MEASUREMENT

            flux = gmix.get_flux() / wcs.pixelArea()  # flux is in ADU.  Should ~ match sum of pixels

        return dx, dy, g1, g2, T, flux, flag



def get_obs_bootstrapper(radius ,usepsf=True, method="fit"):
    import ngmix
    if method=="moments":
        fitter = ngmix.gaussmom.GaussMom(fwhm=radius)
        runner = ngmix.runners.Runner(fitter=fitter, ntry=2)
        if usepsf:
            psf_fitter = ngmix.gaussmom.GaussMom(fwhm=radius)
            psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, ntry=2)
            boot = ngmix.bootstrap.Bootstrapper( runner=runner, psf_runner=psf_runner)
        else:
            boot = ngmix.bootstrap.Bootstrapper(runner=runner)
    if method=="fit":
        ngauss=1
        ngauss_psf=1
                
        wcs=galsim.PixelScale(0.1)
        T=radius
        #rng = galsim.BaseDeviate(1234)
        rng = np.random.RandomState(1234)
        prior = make_ngmix_prior(T, wcs.minLinearScale(),rng)
        guesser = ngmix.guessers.TPSFFluxAndPriorGuesser( rng=rng, T=0.25, prior=prior)
        fitter = ngmix.fitting.CoellipFitter(ngauss=ngauss)
        runner = ngmix.runners.Runner(fitter=fitter,guesser=guesser, ntry=2)
        
        if usepsf:
            psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=ngauss_psf)
            psf_fitter = ngmix.fitting.CoellipFitter(ngauss=ngauss_psf)
            psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=2)
            boot = ngmix.bootstrap.Bootstrapper( runner=runner, psf_runner=psf_runner)
        else:
            boot = ngmix.bootstrap.Bootstrapper(runner=runner)
    return boot

def measure(imgfile, psffile, catalog, xname="x", yname="y", fwhm=None, weight=None,  filename=None,  skystats=True, nmaskedpix=True,aperture_r=20, extra_cols=None, use_weight=True, skipdone=True, rot_pair=False, substractsky=True, edgewidth=5, usepsf=True, method="moments"):
        """
        catalog: catalog of objects of the imgfie to measure
         psf_obs: ngmix observation of the PSF
        fwhm: Guess fwhm for the galaxy prior remember to use image scale
        weight: weight image (segmentation map) to get neighbors features
        use_weight: bool to use or not weight (when given) in the adaptive moments step
        skytstats: flag to include skystats
        nmaskedpix: flag to include number of masked pixels
        aperture_r: radi around xname and yname to check the number of masked pixels
        extra_cols: save cols from catalog 
        """
        starttime = datetime.now()
        if extra_cols is not None:
                extra_cols=[col for col in extra_cols if col in catalog.dtype.names]
                logger.debug("Adding %i extra cols in negmix measure using input catalog information"%(len(extra_cols)))
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
                                    logger.info("Measurement file %s exist and have all existing extra column from sex cat"%(filename))
                                    return
                            else:
                                logger.info("Measurement file exist and there are not extra cols")
                                return
                        except Exception as e:
                                logger.info("Unable to read existing measurement catalog %s, redoing it"%(filename))
                                logger.info(str(e))

        if len(catalog)==0:
            logger.info("Catalog %s have not galaxies. Too noisy?"%(imgfile))
            return
        
        prefix="adamom_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                print("Loading FITS image %s..." % (imgfile))
                logger.info("Loading FITS image %s..." % (imgfile))
                img = galsim.fits.read(imgfile)
                #img.setOrigin(0,0)
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
        except Exception as e:
                logger.error("unable to read psffile %s"%(psffile))
                raise RuntimeError(str(e))

                
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
        #logger.info("Using stampsize of %i"%(stampsize))
        for i, gal in enumerate(catalog):

                flag=0
                        
                #logger.info("Using stampsize %i"%(stampsize))
                        
                (x, y) = (gal[xname], gal[yname])
                pos = galsim.PositionD(x,y)
                
                lowx=int(np.floor(x-0.5*stampsize)) 
                lowy=int(np.floor(y-0.5*stampsize))
                upperx=int(np.floor(x+0.5*stampsize))
                uppery=int(np.floor(y+0.5*stampsize))
                if lowx <1 :flag=BORDER_IMG ;lowx=1
                if lowy <1 :flag=BORDER_IMG ;lowy=1
                if upperx > imagesize : flag=BORDER_IMG; upperx =imagesize
                if uppery > imagesize : flag=BORDER_IMG; uppery =imagesize
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
                                        r_nmpix = aperture_r

                                galseg = np.isin(img_seg_stamp.array,  [img_seg_stamp[center]] )*1
                                galseg_area=np.sum(galseg)
                        
                else:
                        gps_w = None
                        nmaskedpix=False
                        assert ~use_weight

                try:
                    fwhm=3.55
                    flag_ngmix,output,g1,g2,g1_err,g2_err  = ngmix_fit(gps, psfimg, gps_w, fwhm, x, y, usepsf=usepsf, method=method)

                    if flag_ngmix>0:
                        logger.info("ngmix bad measurements")
                        continue
                    hsmparams = galsim.hsm.HSMParams(max_mom2_iter=1000)
                    flag|=flag_ngmix
                    res = galsim.hsm.FindAdaptiveMom(gps,  weight=gps_w, guess_sig=15.0, guess_centroid=pos, hsmparams=hsmparams)
                except:
                    logger.debug("NGMIX with default settings failed on:\n %s" % (str(gal)), exc_info = True)              
                    continue # skip to next stamp !
                                
                ada_flux = res.moments_amp
                ada_x = res.moments_centroid.x + 1.0 # Not fully clear why this +1 is needed. Maybe it's the setOrigin(0, 0).
                ada_y = res.moments_centroid.y + 1.0 # But I would expect that GalSim would internally keep track of these origin issues.
                ada_g1 = res.observed_shape.g1
                ada_g2 = res.observed_shape.g2
                ada_sigma = res.moments_sigma
                ada_rho4 = res.moments_rho4
                
                if ada_flux<0: continue

                
                adamom_list=[ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_sigma, ada_rho4] + list( output.values())+ [ g1,g2,g1_err,g2_err]           

            
                fields_names=[ "adamom_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma", "rho4"] ]+["ngmix_%s_%s%s"%(method,f,"corr"*usepsf) for f in output.keys()]+ ["ngmix_%s_%s%s"%(method,f,"corr"*usepsf) for f in ["g1","g2","g1_err","g2_err"]]
                if skystats:
                        sky_list=[sky["std"], sky["mad"], sky["mean"], sky["med"],sky["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, r_nmpix, galseg_area]+ fracpix + fracpix_md
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
                assert os.path.isfile(filename)


        endtime = datetime.now()        
        logger.info("All done")

        n = len(catalog)
        nfailed = n - len(data)
        
        logger.info("I failed on %i out of %i sources (%.1f percent)" % (nfailed, n, 100.0*float(nfailed)/float(n)))
        logger.info("This measurement took %.3f ms per galaxy" % (1e3*(endtime - starttime).total_seconds() / float(n)))
