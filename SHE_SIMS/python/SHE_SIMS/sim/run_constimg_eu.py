
"""
Running with constant Shear and PSF in the field of view
This should be use for training weights only 
"""

import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
import copy
from . import params_eu_gen
from .. import calc
from .. import utils
import logging
import os
import datetime
import tempfile
import fitsio
from astropy.io import fits
import glob
import multiprocessing
logger = logging.getLogger(__name__)

import galsim
import matplotlib.pyplot as plt
import astropy.table
import momentsml
from momentsml.tools.feature import Feature


profiles=["Gaussian", "Sersic", "EBulgeDisk"]

def drawsourceflagshipcat(sourcecat, catpath,  plotpath=None):
        with fits.open(sourcecat) as hdul:
                cat=hdul[1].data
                cat=pd.DataFrame(cat.astype(cat.dtype.newbyteorder('=')))
                hdul.close()
        print(cat)
        cat["tru_mag"]=-2.5*np.log10(cat["euclid_vis"]) - 48.6

        keepcols=["true_redshift_gal", "tru_mag", "bulge_r50", "bulge_nsersic", "bulge_ellipticity", "disk_r50", "disk_ellipticity", "inclination_angle", "gamma1", "gamma2", "dominant_shape", "disk_angle", "bulge_fraction", "disk_scalelength", "bulge_axis_ratio"]

        cat=cat[keepcols].to_records(index=False)


        #Uncomment for defining S/N
        #flag_mag=(cat["tru_mag"]>24.499)&(cat["tru_mag"]<24.501)
        #flag_rad=(cat["bulge_r50"]>0.5)

        flag_mag=(cat["tru_mag"]<26.0)&(cat["tru_mag"]>20.0)
        flag_rad=(cat["bulge_r50"]<1.2)
        
        cat=cat[flag_mag&flag_rad]

        fitsio.write(catpath, cat, clobber=True)
        

def drawcat(ngal=None, ngal_min=5, ngal_max=20, ngal_nbins=5, nstar=0, nstar_min=5, nstar_max=20, nstar_nbins=5, nimgs=None, ntotal_gal=None,  imagesize=None,  snc=True, mode='grid', tru_type=2, constants=None, dist_type='flagship',sourcecat=None, starsourcecat=None, psf_files=None, usevarpsf=False, sky_vals=None,fixorientation=False, max_shear=0.05, filename=None,  ):
        '''
        ngal: number of galaxies per image
        nstars: number of stars per image
        nimgs: number of images to draw nrealization will be ngal*nimgs  
        ntotal_gal: total number of galaxies per case
        tru_type: type of galaxy 0 (gaussian), 1 (Sersic), 2 (BulgeDisk)
        '''
        logger.info("Drawing catalog")
        if ngal is None:
                ngal = params_eu_gen.draw_ngal(nmin=ngal_min, nmax=ngal_max, nbins=ngal_nbins)
                logger.info("Galaxy density %i"%(ngal))
        if nstar is None:
                nstar = params_eu_gen.draw_ngal(nmin=nstar_min, nmax=nstar_max, nbins=nstar_nbins)
                logger.info("Star density density %i"%(nstar))

        if (nimgs is not None)&(ntotal_gal is not None):
                logger.info("Provide either nimgs or ntotal_gal not both")
                raise

        #ntotal_gal is only an approximation
        # if it is not multiple of ngal(density) then do not include image with the residuo
        # since smaller density
        if ntotal_gal is not None:
                nimgs=int(ntotal_gal/ngal)

        nreas = ngal * nimgs
        nreas_star= nstar*nimgs
        
        if snc: snc_type = nreas
        else: snc_type = 0
        constants["snc_type"]=snc_type
        constants["psf_path"]=os.path.dirname(psf_files[0])

        psf_files=[os.path.basename(a) for a in psf_files]
        profile_type=profiles[tru_type]
        shear_pars= params_eu_gen.draw_s(max_shear=max_shear)
        if usevarpsf:
                psf_file= np.random.choice(psf_files)
        else:
                psf_file= psf_files[0]
                
        if sky_vals is not None:
                sky=np.random.choice(sky_vals)/565.0 #the catalog with skyback is in electros for 565 s exposure
                tru_sky_level=(sky*constants["exptime"])/constants["realgain"]
                constants.update({"sky_level":tru_sky_level})
                constants.update({"skyback":sky})
        else:
                tru_sky_level= (constants["skyback"]*constants["exptime"])/constants["realgain"]
                constants.update({"sky_level":tru_sky_level})

        #Constants
        names_const =  ['tru_gal_density', 'tru_ngals', 'imagesize', 'nimgs', 'tru_type'] +list(shear_pars.keys())+list(constants.keys())+['psf_file']
        formats_const = ['f4', 'i4','i4', 'i4', 'i4'] + [type(shear_pars[k]) for k in shear_pars.keys()]+[type(constants[k]) for k in constants.keys()]+['U100']
        formats_const=[e if (e!=str)&(e!=np.str_) else 'U200' for e in formats_const]
        values_const=[ngal/(imagesize**2),ngal, imagesize, nimgs, tru_type] + [shear_pars[k] for k in shear_pars.keys()]+[constants[k] for k in constants.keys()]+[psf_file]

        #For training point estimates
        x_list=[]; y_list=[]
        if (snc):
                sncrot = 180.0/float(nreas) #rot angle
                logger.info("Drawing a catalog of %i SNC version of the same galaxy distributed on %i images ..." %(nreas, nimgs))

                gal= params_eu_gen.draw(tru_type=tru_type, dist_type=dist_type,  sourcecat=sourcecat, constants=constants)
                x_list, y_list=params_eu_gen.draw_position_sample(nimgs*ngal, imagesize, ngals=ngal, mode=mode)
        else:
                logger.info("Drawing a catalog of %i truly different galaxies distributed on %i images ..." %(nreas, nimgs))
                if (mode=='grid')&(nstar>0):
                        x_list_join, y_list_join=params_eu_gen.draw_position_sample(nreas+nreas_star, imagesize, ngals=ngal+nstar, mode=mode)
                        indx=np.concatenate([np.random.choice(range(i*(ngal+nstar),(i+1)*(ngal+nstar)),ngal, replace=False) for i in range(nimgs)])
                        cindx=np.array(list(set(range(nreas+nreas_star))-set(indx)))
                        x_list=np.array(x_list_join)[indx]
                        y_list=np.array(y_list_join)[indx]
                        x_list_star=np.array(x_list_join)[cindx]
                        y_list_star=np.array(y_list_join)[cindx]
                        #assert False
                else:
                        x_list, y_list=params_eu_gen.draw_position_sample(nreas, imagesize, ngals=ngal, mode=mode)
                        x_list_star, y_list_star=params_eu_gen.draw_position_sample(nreas_star, imagesize, ngals=nstar, mode=mode)

                gal= params_eu_gen.draw_sample(nreas,tru_type=tru_type,  dist_type=dist_type, sourcecat=sourcecat, constants=constants)
                star= params_eu_gen.draw_sample_star(nreas_star, flux_list=starsourcecat, psf_files=psf_files )
                if starsourcecat is None:
                        if nstar==ngal:
                                star["psf_flux"]=gal["tru_flux"]
                                star["psf_mag"]=gal["tru_mag"]
                        else:
                                star= params_eu_gen.draw_sample_star(nreas_star, flux_list=gal["tru_flux"], psf_files=psf_files )
                                
        
        
        #Variable data
        names_var =  ['img_id', 'obj_id',  'x',  'y'] 
        formats_var= ['i4','i4', 'f4' ,'f4']        

                
        if (snc):
                var_gal_names=[ "tru_theta", "tru_g1", "tru_g2"]
                var_gal_formats=[type(gal[f]) for f in var_gal_names]
                const_gal_names=list(set(gal.keys())-set(var_gal_names))
                const_gal_formats=[type(gal[f]) for f in const_gal_names]
                
                dtype_const = dict(names = names_const+const_gal_names,
                                   formats=formats_const+const_gal_formats)
                outdata_const = np.recarray( (1, ), dtype=dtype_const)
                
                dtype_var = dict(names = names_var+var_gal_names,
                                 formats=formats_var+var_gal_formats)
                outdata_var = np.recarray((len(x_list), ), dtype=dtype_var)
                
                        
                for key in const_gal_names:
                        outdata_const[key] = gal[key]
                if ("tru_theta" in gal.keys()):
                        outdata_var["tru_theta"] = gal['tru_theta'] +np.array([ k*sncrot for k in range(nreas)])
                if ("tru_g1" in gal.keys()) & ("tru_g2" in gal.keys()):
                        outdata_var["tru_g1"] = np.array([ calc.rotg(gal["tru_g1"] ,gal["tru_g2"],k*sncrot)[0] for k in range(nreas)])
                        outdata_var["tru_g2"] = np.array([ calc.rotg(gal["tru_g1"] ,gal["tru_g2"],k*sncrot)[1] for k in range(nreas)])

        else:
                dtype_const = dict(names = names_const, formats=formats_const)
                outdata_const = np.recarray( (1, ), dtype=dtype_const)

                gal_names=list(gal.keys())
                gal_formats=[type(gal[g][0]) for g in gal_names]
                dtype_var = dict(names = names_var+gal_names,
                                 formats=formats_var+gal_formats)
                outdata_var = np.recarray((len(x_list), ), dtype=dtype_var)
                
                for key in gal_names:
                        outdata_var[key] = gal[key]

                if nstar>0:
                        star_names=list(star.keys())
                        star_formats=[type(star[g][0]) for g in star_names]
                        star_formats=[e if (e!=str)&(e!=np.str_) else 'U100' for e in star_formats]
                        dtype_star = dict(names = names_var+star_names,
                                          formats=formats_var+star_formats)
                        outdata_star = np.recarray((len(x_list_star), ), dtype=dtype_star)

                        for key in star_names:
                                print(key, len(outdata_star[key]), len(star[key]))
                                outdata_star[key] = star[key]
                                assert len(star[key])==len(x_list_star)
                        outdata_star["x"]=x_list_star
                        outdata_star["y"]=y_list_star
                        outdata_star['img_id']=np.concatenate([ [k]*nstar for k in range(nimgs) ])
                        outdata_star['obj_id']=[i for i in range(nstar)]*nimgs

        if fixorientation:
                indx=np.random.choice(range(nreas))
                outdata_var["tru_g1"]=[outdata_var["tru_g1"][indx]]*nreas
                outdata_var["tru_g2"]=[outdata_var["tru_g2"][indx]]*nreas
                outdata_var["tru_theta"]=[outdata_var["tru_theta"][indx]]*nreas

        for i, key in enumerate(names_const):
                outdata_const[key] = values_const[i]
        outdata_var["x"]=x_list
        outdata_var["y"]=y_list
        outdata_var['img_id']=np.concatenate([ [k]*ngal for k in range(nimgs) ])
        outdata_var['obj_id']=[i for i in range(ngal)]*nimgs
        

        if filename is not None:
                fitsio.write(filename, outdata_var, clobber=True)
                #Write constants in the second header
                fitsio.write(filename, outdata_const, clobber=False)
                if nstar>0:
                        fitsio.write(filename, outdata_star, clobber=False)      
                logger.info("catalog %s written"%(filename))

       
def drawimg(catalog, const_cat, filename, starcatalog=None, psfimg=True, gsparams=None, sersiccut=None, savetrugalimg=False, savetrupsfimg=False, rot_pair=False, pixel_conv=False, constantshear=True):
        '''
        constantshear: flag to determine whether or not there is a variable shear in a image
        '''

        starttime = datetime.datetime.now()
        logger.info("Drawing image %s"%(filename))
        
        if rot_pair:
                # this only with the porpouse of validation with shape noise cancellation for training should not be used
                logger.info("Creating images with galaxies rotated 90 degrees")
                sncrot=90
                aux1,aux2=np.vectorize(calc.rotg)(catalog["tru_g1"],catalog["tru_g2"],sncrot)
                catalog["tru_g1"]=aux1
                catalog["tru_g2"]=aux2
                        
        
                
        vispixelscale=0.1 #arcsec
        psfpixelscale = 0.02 #arcsec
        if gsparams is None:
                gsparams = galsim.GSParams(maximum_fft_size=50000)

        if "imagesize" not in const_cat.dtype.names:
                raise RuntimeError("Not imagesize found in const_cat")
        else:
                imagesize=const_cat["imagesize"][0]
        if "tru_pixel" in const_cat.dtype.names:
                tru_pixel = const_cat["tru_pixel"][0]
                if tru_pixel > 0:
                        logger.warning("The galaxy profiles will be convolved with an extra {0:.1f} pixels".format(tru_pixel))
                        pix = galsim.Pixel(const_cat["tru_pixel"][0])
                else:
                        pix = None
                
        # Galsim random number generators
        rng = galsim.BaseDeviate()
        ud = galsim.UniformDeviate()
        gal_image = galsim.ImageF(imagesize , imagesize)

        if savetrugalimg:
                tru_gal_image = galsim.ImageF(imagesize , imagesize)

        if savetrupsfimg:
                tru_psf_image = galsim.ImageF(imagesize , imagesize)
        
        #Loading the constant PSF
        if psfimg:
                #logger.info("Reading psf file %s"%(row["psf_file"]))
                #psfarray=fitsio.read(const_cat["psf_file"][0])
                psffilename=os.path.join(const_cat["psf_path"][0],const_cat["psf_file"][0])
                assert os.path.isfile(psffilename)
                with fits.open(psffilename) as hdul:
                        psfarray=hdul[1].data
                psfarray_shape=psfarray.shape
                psfimg=galsim.Image(psfarray, xmin=0,ymin=0)
                #psfimg = galsim.fits.read(row["psf_file"])
                psfimg.setOrigin(0,0)

                psf = galsim.InterpolatedImage( psfimg, flux=1.0, scale=psfpixelscale, gsparams=gsparams )
                psf.image.setOrigin(0,0)

        else:
                assert pixel_conv
 

                logger.info("Using Gaussian PSF")
                psf = galsim.Gaussian(flux=1., sigma=3.55*psfpixelscale)
                psf = psf.shear(g1=-0.0374, g2=9.414e-06 )

        
        for row in catalog:
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

                        '''
                        bulge_ell = galsim.Shear(g=tru_bulge_g, beta=tru_theta * galsim.degrees)
                        bulge = bulge.shear(bulge_ell)
                        gal=bulge
                        
                        if dominant_shape==1:
                                # Get a disk
                                
                                disk = galsim.InclinedExponential(inclination=tru_disk_inclination* galsim.degrees , 
                                                                  half_light_radius=tru_disk_rad,
                                                                  flux=tru_disk_flux, 
                                                                  scale_height=tru_disk_scaleheight,
                                                                  gsparams = gsparams)
                                disk = disk.rotate(tru_theta* galsim.degrees)                
                                gal += disk
                        '''

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

                # And now we add lensing, if s1, s2 and mu are different from no lensing...
                if constantshear:
                        if const_cat["tru_s1"][0] != 0 or const_cat["tru_s2"][0] != 0 or const_cat["tru_mu"][0] != 1:
                                gal = gal.lens(float(const_cat["tru_s1"][0]), float(const_cat["tru_s2"][0]), float(const_cat["tru_mu"][0]))
                        else: pass
                else:
                       if row["tru_s1"]!= 0 or row["tru_s2"] != 0 or row["tru_mu"] != 1:
                                gal = gal.lens(float(row["tru_s1"]), float(row["tru_s2"]), float(row["tru_mu"]))
                       else: pass 

                xjitter = vispixelscale*(ud() - 0.5) # This is the minimum amount -- should we do more, as real galaxies are not that well centered in their stamps ?
                yjitter = vispixelscale*(ud() - 0.5)
                gal = gal.shift(xjitter,yjitter)

                
                # CONVOLUTION WITH THE PSF           
                if pix is not None: # Not sure if this should only apply to gaussian PSFs, but so far this seems OK.
                        # Remember that this is an "additional" pixel convolution, not the usual sampling-related convolution that happens in drawImage.
                        galconv = galsim.Convolve([gal, psf, pix])
                        
                else:
                        galconv = galsim.Convolve([gal,psf])

                # Taken from Galsim's demo11:
                x_nominal = row["x"] + 0.5
                y_nominal = row["y"] + 0.5
                ix_nominal = int(np.floor(x_nominal+0.5))
                iy_nominal = int(np.floor(y_nominal+0.5))
                dx = x_nominal - ix_nominal
                dy = y_nominal - iy_nominal
                offset = galsim.PositionD(dx,dy)
                
                if pixel_conv:
                        stamp = galconv.drawImage(offset=offset, scale=vispixelscale)
                else:
                        # This sould be used for psf image (interpolated) that already include the pixel response
                        # if method auto were used extra convolution will be included
                        stamp = galconv.drawImage(offset=offset, method="no_pixel", scale=psfpixelscale)
                        nbins=int(vispixelscale/psfpixelscale)
                        #logger.info("rebinning stamp using %ix%i grid"%(nbins,nbins))
                        stamp= stamp.bin(nbins,nbins)
                        
                stamp.setCenter(ix_nominal,iy_nominal)
                
                bounds = stamp.bounds & gal_image.bounds
                if not bounds.isDefined():
                        logger.info("Out of bounds")
                        continue
                
                gal_image[bounds] += stamp[bounds]

                #vispixelscale=None
                if savetrugalimg:
                        tru_gal_stamp=gal.drawImage(offset=offset, scale=vispixelscale)
                        tru_gal_stamp.setCenter(ix_nominal,iy_nominal)
                        bounds = tru_gal_stamp.bounds & tru_gal_image.bounds
                        tru_gal_image[bounds]+= tru_gal_stamp[bounds]
                if savetrupsfimg:
                        if pixel_conv:
                                tru_psf_stamp=psf.drawImage(offset=offset, scale=vispixelscale)
                                tru_psf_stamp.setCenter(ix_nominal,iy_nominal)
                                bounds = tru_psf_stamp.bounds & tru_psf_image.bounds
                                tru_psf_image[bounds]+= tru_psf_stamp[bounds]
                        else:
                                tru_psf_stamp=psf.drawImage(offset=offset, method="no_pixel", scale=psfpixelscale)
                                nbins=int(vispixelscale/psfpixelscale)
                                tru_psf_stamp= tru_psf_stamp.bin(nbins,nbins)
                                tru_psf_stamp.setCenter(ix_nominal,iy_nominal)
                                bounds = tru_psf_stamp.bounds & tru_psf_image.bounds
                                tru_psf_image[bounds]+= tru_psf_stamp[bounds]


        if starcatalog is not None:
                for row in starcatalog:
                        '''
                        if pixel_conv: 
                                logger.info("Using PSF core image without pixel response")
                        else:
                                logger.info("Using PSF core image with pixel response")
                        '''

                        psffilename=os.path.join(const_cat["psf_path"][0],row["psf_file"])
                        assert os.path.isfile(psffilename)
                        with fits.open(psffilename) as hdul:
                                psfarray=hdul[1].data
                        psfarray_shape=psfarray.shape
                        psfimg=galsim.Image(psfarray, xmin=0,ymin=0)
                        psfimg.setOrigin(0,0)

                        psf = galsim.InterpolatedImage( psfimg, flux=row["psf_flux"], scale=psfpixelscale, gsparams=gsparams )
                        psf.image.setOrigin(0,0)

                        xjitter = vispixelscale*(ud() - 0.5) 
                        yjitter = vispixelscale*(ud() - 0.5)
                        psf = psf.shift(xjitter,yjitter)
                        x_nominal = row["x"] + 0.5
                        y_nominal = row["y"] + 0.5
                        ix_nominal = int(np.floor(x_nominal+0.5))
                        iy_nominal = int(np.floor(y_nominal+0.5))
                        dx = x_nominal - ix_nominal
                        dy = y_nominal - iy_nominal
                        offset = galsim.PositionD(dx,dy)

                        stamp = psf.drawImage(offset=offset, scale=vispixelscale)
                        stamp.setCenter(ix_nominal,iy_nominal)
                        bounds = stamp.bounds & gal_image.bounds
                        if not bounds.isDefined():
                                logger.info("Out of bounds")
                                continue
                        gal_image[bounds] += stamp[bounds]
                
                gal_image[bounds] += stamp[bounds]
                if savetrugalimg:
                        tru_gal_stamp=psf.drawImage(offset=offset, scale=vispixelscale)
                        tru_gal_stamp.setCenter(ix_nominal,iy_nominal)
                        bounds = tru_gal_stamp.bounds & tru_gal_image.bounds
                        tru_gal_image[bounds]+= tru_gal_stamp[bounds]
                        
                        

                                
        # And add noise to the convolved galaxy:
        if savetrugalimg:
                logger.info("Saving image without noise")
                filename_nf=filename.replace("_galimg.fits","_trugalimg.fits")
                if rot_pair: filename_nf=filename.replace("_galimg_rot.fits","_trugalimg_rot.fits")
                tru_gal_image.write(filename_nf)
        if savetrupsfimg:
                logger.info("Saving psf image without noise")
                filename_nf=filename.replace("_galimg.fits","_trupsfimg.fits")
                if rot_pair: filename_nf=filename.replace("_galimg_rot.fits","_trugalimg_rot.fits")
                tru_psf_image.write(filename_nf)
        
                
        # And add noise to the convolved galaxy:
        
        #gal_image.addNoise(galsim.CCDNoise(rng,
        #                                   sky_level=float(const_cat["tru_sky_level"][0]),
        #                                   gain=float(const_cat["tru_gain"][0]),
        #                                   read_noise=float(const_cat["tru_read_noise"][0])))

        gal_image+=float(const_cat["sky_level"][0])
        gal_image.addNoise(galsim.CCDNoise(rng,
                                           sky_level=0.0,
                                           gain=float(const_cat["realgain"][0]),
                                           read_noise=float(const_cat["ron"][0])))
                
        logger.info("Done with drawing, now writing output FITS files ...")        

        gal_image.write(filename)
        

def makeworkerlist(workdir, catalogs, basename_list, drawimgkwargs, skipdone, ext='_galimg.fits', strongcheck=False):
        wslist=[]
        for cat,name in zip(catalogs,basename_list):
                assert os.path.isfile(cat)
                with fits.open(cat) as hdul:
                        hdul.verify('fix')
                        print(cat)
                        print(hdul)
                        constcat=hdul[2].data
                nimgs=constcat['nimgs'][0]
                for img_id in range(nimgs):
                        imgdir=os.path.join(workdir,"%s_img"%(name))
                        if not os.path.exists(imgdir):
                                os.makedirs(imgdir)
                                logger.info("Creating a new set of simulations named '%s'" % (imgdir))
                        else:
                                logger.info("Checking/Adding new simulations to the existing set '%s'" % (imgdir))
                        imgname=os.path.join(imgdir,"%s_img%i%s"%(name,img_id,ext))
                        
                        ws = _WorkerSettings(cat, img_id, imgname, drawimgkwargs)
                        if skipdone:
                                if os.path.isfile(imgname):
                                        if strongcheck:
                                                try:
                                                        with fits.open(imgname) as hdul:
                                                                hdul.verify('fix')
                                                                img=hdul[0].data
                                                        continue
                                                except Exception as e:
                                                        logger.info("Unable to read image %s redoing"%(imgname))
                                                        logger.info(str(e))
                                        else:
                                                if not os.path.getsize(imgname):
                                                        logger.info("Redoing %s, it is empty"%(imgname))
                                                        os.remove(imgname)
                                                else:
                                                        continue
                                else:
                                        logger.info("Adding sim file %s "%(imgname))

                        wslist.append(ws)
        return wslist


def multi(workdir, drawcatkwargs, drawimgkwargs, ncat=1, ncpu=1, skipdone=False, rot_pair=False, strongcheck=False):
        ''''
        rot_pair: make rotated pair imgs
        '''
        
        if not os.path.exists(workdir):
                os.makedirs(workdir)
                logger.info("Creating a new set of simulations named '%s'" % (workdir))
        else:
                logger.info("Adding new simulations to the existing set '%s'" % (workdir))

        starttime = datetime.datetime.now()
        prefix = starttime.strftime("%Y%m%dT%H%M%S_")
        logger.info("Drawing catalogs")
        catalogs=[]
        basename_list=[]

        if skipdone:
                catalogs= sorted(glob.glob(os.path.join(workdir, "*_cat.fits")))
                basenames=[ os.path.basename(cat).replace("_cat.fits","") for cat in catalogs ]
                basename_list+=basenames
                ncat-=len(catalogs)

                logger.info("Adding %i new catalogs in '%s'" % (ncat, workdir))
        
        for i in range(ncat):
                catfile = tempfile.NamedTemporaryFile(mode='wb', prefix=prefix, suffix="_cat.fits", dir=workdir, delete=False)
                catalogs.append(str(catfile.name))
                basename=os.path.basename(str(catfile.name)).replace("_cat.fits","")
                basename_list.append(basename)
                constantshear=drawimgkwargs["constantshear"]
                if constantshear:
                        drawcat(**drawcatkwargs, filename=str(catfile.name))
                else:
                        drawcat_varshear(**drawcatkwargs, filename=str(catfile.name))
        logger.info("Drawing true catalogs finished")

        
        if rot_pair:
                drawimgkwargs.update({"rot_pair":True})
                wslist= makeworkerlist(workdir, catalogs, basename_list, drawimgkwargs, skipdone, ext='_galimg_rot.fits', strongcheck=strongcheck)
                _run(wslist, ncpu)
                
        drawimgkwargs.update({"rot_pair":False})
        wslist= makeworkerlist(workdir, catalogs, basename_list, drawimgkwargs, skipdone,  ext='_galimg.fits', strongcheck=strongcheck)
        _run(wslist, ncpu)

                        
        
class _WorkerSettings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, cat_file, img_id, filename, drawimgkwargs ):
                
                self.cat_file= cat_file
                self.img_id = img_id
                self.filename=filename
                self.drawimgkwargs=drawimgkwargs
                

def _worker(ws):
        """
        Worker function that the different processes will execute, processing the
        _SexWorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed() #this is important
        p = multiprocessing.current_process()
        logger.info("%s is starting measure catalog %s with PID %s" % (p.name, str(ws), p.pid))


        data=fitsio.read(ws.cat_file)
        cat=data[data["img_id"]==ws.img_id]
        const=fitsio.read(ws.cat_file, ext=2)
        try:
                stardata=fitsio.read(ws.cat_file, ext=3)
                starcat=stardata[stardata["img_id"]==ws.img_id]
        except:
                starcat=None
        drawimg(cat, const, ws.filename, starcatalog=starcat, **ws.drawimgkwargs )
        
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))


def _run(wslist, ncpu):
        """
        Wrapper around multiprocessing.Pool with some verbosity.
        """
        
        if len(wslist) == 0: # This test is useful, as pool.map otherwise starts and is a pain to kill.
                logger.info("No images to measure.")
                return

        if ncpu == 0:
                try:
                        ncpu = multiprocessing.cpu_count()
                except:
                        logger.warning("multiprocessing.cpu_count() is not implemented!")
                        ncpu = 1
        
        starttime = datetime.datetime.now()
        
        logger.info("Starting the drawing of %i images using %i CPUs" % (len(wslist), ncpu))
        
        if ncpu == 1: # The single process way (MUCH MUCH EASIER TO DEBUG...)
                list(map(_worker, wslist))
        
        else:
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(_worker, wslist)
                pool.close()
                pool.join()
        
        endtime = datetime.datetime.now()
        logger.info("Done, the total running time was %s" % (str(endtime - starttime)))



def transformtogrid(inputdir, outputdir, drawcatkwargs):
        catalogs= sorted(glob.glob(os.path.join(inputdir, "*_cat.fits")))
        assert len(catalogs)>0
        outcatalogs=[os.path.join(outputdir, os.path.basename(cat).replace("_cat.fits","grid_cat.fits")) for cat in catalogs]
        for icat, ocat in zip(catalogs,outcatalogs):
                fromrandomtogrid(icat, drawcatkwargs, outputcatname=ocat)
                
        
def fromrandomtogrid(inputcat, drawcatkwargs, outputcatname=None):
        logger.info("Replacing in a grid %s"%(inputcat))
        incat=fitsio.read(inputcat)
        incat_df = pd.DataFrame(incat.astype(incat.dtype.newbyteorder('=')))
        constcat=fitsio.read(inputcat, ext=2)
        constcat_df=pd.DataFrame(constcat.astype(constcat.dtype.newbyteorder('=')))
        nimgs=constcat['nimgs'][0]

        ngal= drawcatkwargs['ngal']
        imagesize=drawcatkwargs['imagesize']
        assert ngal is not None
        assert imagesize is not None

        ntotalgal=len(incat)
        nimgs=ntotalgal//ngal
        res=ntotalgal%ngal
        if res>0:
                nimgs+=1
                x_list, y_list=params_eu_gen.draw_position_sample(ngal*nimgs, imagesize, ngals=ngal, mode='grid')
                x_list=x_list[:ntotalgal]
                y_list=y_list[:ntotalgal]
                imgids=np.concatenate([[i]*ngal for i in range(nimgs)])[:ntotalgal]
        else:
                x_list, y_list=params_eu_gen.draw_position_sample(ntotalgal, imagesize, ngals=ngal, mode='grid')
                imgids=np.concatenate([[i]*ngal for i in range(nimgs)])
                
        incat_df['img_id']=imgids
        incat_df['obj_id']=np.concatenate([ list(range(len(incat_df[incat_df['img_id']==img_id]))) for img_id in set(incat_df['img_id'])])
        incat_df['x']=x_list
        incat_df['y']=y_list
        constcat_df['nimgs']=[nimgs]
        
        if outputcatname is not None:
                fitsio.write(outputcatname, incat_df.to_records(index=False), clobber=True)
                #Write constants in the second header
                fitsio.write(outputcatname, constcat_df.to_records(index=False), clobber=False)
                logger.info("catalog %s written"%(outputcatname))



def update_psf_sky(inputdir, outputdir, drawcatkwargs):
        catalogs= sorted(glob.glob(os.path.join(inputdir, "*_cat.fits")))
        assert len(catalogs)>0
        outcatalogs=[os.path.join(outputdir, os.path.basename(cat).replace("_cat.fits","update_cat.fits")) for cat in catalogs]
        for icat, ocat in zip(catalogs,outcatalogs):
                incat=fitsio.read(icat)
                incat_df = pd.DataFrame(incat.astype(incat.dtype.newbyteorder('=')))
                constcat=fitsio.read(icat, ext=2)
                constcat_df=pd.DataFrame(constcat.astype(constcat.dtype.newbyteorder('=')))

                psf_files=drawcatkwargs['psf_files']
                sky_vals=drawcatkwargs['sky_vals']
                const_type=drawcatkwargs['const_type']
                if len(psf_files)==1: constcat_df['psf_file']=psf_files
                if sky_vals is None:
                        constants= params_eu_gen.constants(snc_type=constcat_df['snc_type'], const_type=const_type, tru_type=constcat_df['tru_type'])
                        constcat_df['tru_sky_level']=[constants["tru_sky_level"]]
        
                if ocat is not None:
                        fitsio.write(ocat, incat_df.to_records(index=False), clobber=True)
                        #Write constants in the second header
                        fitsio.write(ocat, constcat_df.to_records(index=False), clobber=False)
                        logger.info("catalog %s written"%(ocat))
   
