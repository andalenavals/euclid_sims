
"""
Running with constant Shear and PSF in the field of view
This should be use for training weights only 
"""

import numpy as np
from numpy.lib.recfunctions import append_fields
import copy
import pandas as pd
from . import params_eu_gen
from .. import calc
from ..utils import open_fits, _run
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


profiles=["Gaussian", "Sersic", "EBulgeDisk", "CosmosReal", "CosmosParam"]

def drawsourceflagshipcat(sourcecat, catpath,  plotpath=None):
        cat=open_fits(sourcecat)
        cat["tru_mag"]=-2.5*np.log10(cat["euclid_vis"]) - 48.6

        keepcols=["true_redshift_gal", "tru_mag", "bulge_r50", "bulge_nsersic", "bulge_ellipticity", "disk_r50", "disk_ellipticity", "inclination_angle", "gamma1", "gamma2", "dominant_shape", "disk_angle", "bulge_fraction", "disk_scalelength", "bulge_axis_ratio", "euclid_vis"]

        cat=cat[keepcols].to_records(index=False)


        #Uncomment for defining S/N
        #flag_mag=(cat["tru_mag"]>24.499)&(cat["tru_mag"]<24.501)
        #flag_rad=(cat["bulge_r50"]>0.5)

        flag_mag=(cat["tru_mag"]<26.0)&(cat["tru_mag"]>20.0)
        flag_rad=(cat["bulge_r50"]<1.2)
        
        cat=cat[flag_mag&flag_rad]

        fitsio.write(catpath, cat, clobber=True)
        

def drawcat(ngal=None, ngal_min=5, ngal_max=20, ngal_nbins=5, nstar=0, nstar_min=5, nstar_max=20, nstar_nbins=5, nimgs=None, ntotal_gal=None,  imagesize=None,  snc=True, mode='grid', tru_type=2, constants=None, dist_type='flagship',sourcecat=None, starsourcecat=None, psfsourcecat=None, usevarpsf=False, sky_vals=None,fixorientation=False, max_shear=0.05, filename=None, scalefactor=1.0, scalefield=None  ):
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
        assert ngal>0
        if nstar is None:
                nstar = params_eu_gen.draw_ngal(nmin=nstar_min, nmax=nstar_max, nbins=nstar_nbins)
                logger.info("Star density density %i"%(nstar))
                constants["tru_nstars"]=nstar

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
        
        profile_type=profiles[tru_type]
        shear_pars= params_eu_gen.draw_s(max_shear=max_shear)

        
        psfdata=fitsio.read(psfsourcecat)
        psfdataconst=fitsio.read(psfsourcecat, ext=2)
        constants["psf_path"]=psfdataconst["psf_path"][0]
        if usevarpsf:
                psfinfo= list(np.random.choice(psfdata))
        else:
                psfinfo= list(psfdata[0])

        psfnames=list(psfdata.dtype.names)
        psfformats=[ e[1] for e in psfdata.dtype.descr]

        
        if sky_vals is not None:
                sky=np.random.choice(sky_vals)/565.0 #the catalog with skyback is in electros for 565 s exposure
                tru_sky_level=(sky*constants["exptime"])/constants["realgain"]
                if scalefield=='sky_level':
                        tru_sky_level*=scalefactor
                constants.update({"sky_level":tru_sky_level})
                constants.update({"skyback":sky})
        else:
                tru_sky_level= (constants["skyback"]*constants["exptime"])/constants["realgain"]
                if 'scalefield'=='sky_level':
                        tru_sky_level*=scalefactor
                constants.update({"sky_level":tru_sky_level})
   
                
        #Constants
        names_const =  ['tru_gal_density', 'tru_ngals', 'imagesize', 'nimgs', 'tru_type'] +list(shear_pars.keys())+list(constants.keys())+psfnames
        formats_const = ['f4', 'i4','i4', 'i4', 'i4'] + [type(shear_pars[k]) for k in shear_pars.keys()]+[type(constants[k]) for k in constants.keys()]+psfformats
        formats_const=[e if (e!=str)&(e!=np.str_) else 'U200' for e in formats_const]
        values_const=[ngal/(imagesize**2),ngal, imagesize, nimgs, tru_type] + [shear_pars[k] for k in shear_pars.keys()]+[constants[k] for k in constants.keys()]+psfinfo

        #For training point estimates
        x_list=[]; y_list=[]
        if (snc)&(ngal>0):
                sncrot = 180.0/float(nreas) #rot angle
                logger.info("Drawing a catalog of %i SNC version of the same galaxy distributed on %i images ..." %(nreas, nimgs))

                gal= params_eu_gen.draw(tru_type=tru_type,
                                        dist_type=dist_type, sourcecat=sourcecat,
                                        constants=constants, scalefactor=scalefactor,
                                        scalefield=scalefield)
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
                        if (nstar>0): x_list_star, y_list_star=params_eu_gen.draw_position_sample(nreas_star, imagesize, ngals=nstar, mode=mode)

                gal=params_eu_gen.draw_sample(nreas,tru_type=tru_type,
                                          dist_type=dist_type, sourcecat=sourcecat,
                                          constants=constants, scalefactor=scalefactor, scalefield=scalefield)
                if (nstar>0):
                        assert os.path.isfile(starsourcecat)
                        star= params_eu_gen.draw_sample_star(nreas_star, sourcecat=starsourcecat, constants=constants )
        
                if nstar==ngal:
                        star["tru_flux"]=gal["tru_flux"]
                        star["tru_mag"]=gal["tru_mag"]
             

                #print(star)
                #assert False
        
        
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
                if ngal>0:
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

        if ngal>0:
                outdata_var["x"]=x_list
                outdata_var["y"]=y_list
                outdata_var['img_id']=np.concatenate([ [k]*ngal for k in range(nimgs) ])
                outdata_var['obj_id']=[i for i in range(ngal)]*nimgs

        if filename is not None:
                hdul=[fits.PrimaryHDU(header=fits.Header())]
                hdul.insert(1, fits.BinTableHDU(outdata_var))
                hdul.insert(2, fits.BinTableHDU(outdata_const))
                if nstar>0: hdul.insert(3, fits.BinTableHDU(outdata_star))
                hdulist = fits.HDUList(hdul)
                hdulist.writeto(filename, overwrite=True)
                logger.info("catalog %s written"%(filename))


       
def drawimg(catalog, const_cat, filename, starcatalog=None, psfimg=True, gsparams=None, sersiccut=None, savetrugalimg=False, savetrupsfimg=False, rot_pair=False, pixel_conv=False, constantshear=True, cosmoscatfile=None, cosmosdir=None, tru_type=None):
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
                catalog["tru_theta"]+=sncrot
                        
        
                
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
                

        profile_type=profiles[const_cat["tru_type"][0]]
        if tru_type is not None:
                logger.debug("Warning you are changing the profile type to %s"%(profiles[tru_type]))
                profile_type=profiles[tru_type]
                        
        if profile_type == "CosmosReal":
                cosmospath=os.path.join(cosmosdir, cosmoscatfile)
                galaxy_catalog = galsim.COSMOSCatalog(cosmospath, exclusion_level='none', exptime=float(const_cat["exptime"][0]), area=9926)
        elif profile_type == "CosmosParam":
                cosmospath=os.path.join(cosmosdir, cosmoscatfile)
                galaxy_catalog = galsim.COSMOSCatalog(cosmospath, exclusion_level='none', use_real='False', exptime=float(const_cat["exptime"][0]), area=9926)
        
        #print(galaxy_catalog)
        #print(dir(galaxy_catalog))
        #print(type(galaxy_catalog))
        #print(galaxy_catalog.param_cat.dtype)
     
        for row in catalog:
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
                                        
                                gal = gal.rotate(tru_theta* galsim.degrees)
                
                elif profile_type == "CosmosReal":
                        '''
                        exclusion_level='none' includes all images, no internal selection done by COSMOSCatalog.
                        Leaving out this parameter results in ~81k galaxies used instead of all 87k.
                        Refer to: http://galsim-developers.github.io/GalSim/_build/html/real_gal.html#galsim.COSMOSCatalog
                                                
                        exptime and area are the exposure time and effective collecting area for Euclid.
                        Refer to: http://galsim-developers.github.io/GalSim/_build/html/real_gal.html#galsim.GalaxySample.makeGalaxy
                        '''
                        if const_cat["snc_type"][0]==0:
                                index=int(row['cosmos_index'])
                        else:
                                index=int(const_cat['cosmos_index'][0])
                        gal = galaxy_catalog.makeGalaxy(index, gal_type='real', noise_pad_size=90.6*vispixelscale)
                        if align_cosmos:
                            sersic_values = galaxy_catalog.getValue('sersicfit', index)
                            sersic_pa_rad = float(sersic_values[7])
                            sersic_pa = sersic_pa_rad * (180 / np.pi)                            
                            tru_theta = float(row["tru_theta"])
                            rot_angle =  tru_theta - sersic_pa
                            if random_rot_cosmos:
                                rot_180 = np.random.choice([True,False])
                                if rot_180:
                                    rot_angle=rot_angle+180
                                #print(rot_180, (rot_angle-(tru_theta-sersic_pa))) #Should print (True, 180) or (False, 0) if working correctly. 
                            gal = gal.rotate(rot_angle * galsim.degrees)                 
                                        
                elif profile_type == "CosmosParam":
                        
                        '''
                        use_real=False prevents drawing images of real galaxies. Not necessary but used for optimization purposes.
                        '''
                         if const_cat["snc_type"][0]==0:
                                index=int(row['cosmos_index'])
                        else:
                                index=int(const_cat['cosmos_index'][0])
                                              
                        try:
                                """
                                if any([ val >1.5 for val in galaxy_catalog.getValue('hlr', index)[1:]]):
                                        logger.debug("Warning extremely large galaxy founded %i!!"%(index))
                                        continue
                                if galaxy_catalog.getValue('mag_auto', index) < 19:
                                        logger.debug("Warning extremely bright galaxy %i!!"%(index))
                                        continue
                                """
                                gal = galaxy_catalog.makeGalaxy(index, gal_type='parametric')
                                
                        except Exception as e:
                                print(f"An error occurred:{e}")
                                logger.info("Could not make parametric galaxy!!")
                                continue
                        if align_cosmos:
                            sersic_values = galaxy_catalog.getValue('sersicfit', index)
                            sersic_pa_rad = float(sersic_values[7])
                            sersic_pa = sersic_pa_rad * (180 / np.pi)                            
                            tru_theta = float(row["tru_theta"])
                            rot_angle =  tru_theta - sersic_pa
                            if random_rot_cosmos:
                                rot_180 = np.random.choice([True,False])
                                if rot_180:
                                    rot_angle=rot_angle+180
                                #print(rot_180, (rot_angle-tru_theta+sersic_pa)) #Should print (True, 180) or (False, 0) if working correctly. 
                            gal = gal.rotate(rot_angle * galsim.degrees)
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

                try:
                        # Taken from Galsim's demo11:
                        x_nominal = row["x"] + 0.5
                        y_nominal = row["y"] + 0.5
                        ix_nominal = int(np.floor(x_nominal+0.5))
                        iy_nominal = int(np.floor(y_nominal+0.5))
                        dx = x_nominal - ix_nominal
                        dy = y_nominal - iy_nominal
                        offset = galsim.PositionD(dx,dy)

               
                        if pixel_conv:
                                if (profile_type == "CosmosReal"):    #stampsize=64
                                        x_lower = int(row["x"] - 0.5*64)+1
                                        x_upper = int(row["x"] + 0.5*64)
                                        y_lower = int(row["y"] - 0.5*64)+1
                                        y_upper = int(row["y"] + 0.5*64)
                                        sub_image_bounds = galsim.BoundsI(x_lower, x_upper, y_lower, y_upper)                                      
                                        sub_image_bounds = sub_image_bounds & gal_image.bounds
                                        sub_image = gal_image[sub_image_bounds]
                                        stamp = galconv.drawImage(image = sub_image,add_to_image=True, offset=offset, scale=vispixelscale)
                                else:
                                	    stamp = galconv.drawImage(offset=offset, scale=vispixelscale)
                        else:
                                # This sould be used for psf image (interpolated) that already include the pixel response
                                # if method auto were used extra convolution will be included
                                stamp = galconv.drawImage(offset=offset, method="no_pixel", scale=psfpixelscale)
                                nbins=int(vispixelscale/psfpixelscale)
                                #logger.info("rebinning stamp using %ix%i grid"%(nbins,nbins))
                                stamp= stamp.bin(nbins,nbins)
                
                except Exception as e:
                        print(f"An error occurred:{e}")
                        logger.info("Could not draw the image")
                        continue

                        
                stamp.setCenter(ix_nominal,iy_nominal)
                
                bounds = stamp.bounds & gal_image.bounds
                if not bounds.isDefined():
                        logger.info("Out of bounds")
                        continue
                
                if not (profile_type == "CosmosReal"):
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
                logger.debug("Including STARS in sims")
                stampsize=1024
                for row in starcatalog:
                        #psf = galsim.InterpolatedImage( psfimg, flux=row["star_flux"], scale=psfpixelscale, gsparams=gsparams )
                        #psf.image.setOrigin(0,0)
                        psf=psf.withFlux(row["tru_flux"])
                        
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

                        stamp = psf.drawImage(nx=stampsize, ny=stampsize, offset=offset, scale=vispixelscale)
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
        if profile_type == "CosmosReal":
            # Correction: 
            # Step 1: Measure mean skymad on COSMOS Real branch with no added skylevel or CCD noise. Measured skymad = 1.2325080832761701
            # Step 2: Square this value, then multiply by the realgain. Squared mean skymad = 1.5190761753410986
            # Step 3: Subtract final value from skylevel before applying to image.
            # Adjusted skylevel = skylevel - 1.5190761753410986 * realgain
            gal_image+=(float(const_cat["sky_level"][0]) - 1.5190761753410986*float(const_cat["realgain"][0]))
            #gal_image+=(float(const_cat["sky_level"][0]) - 6.0)
        else:
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
                                                        #os.remove(imgname)
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
                drawimgkwargs.update({"rot_pair":True, "tru_type":drawcatkwargs["tru_type"]})
                wslist= makeworkerlist(workdir, catalogs, basename_list, drawimgkwargs, skipdone, ext='_galimg_rot.fits', strongcheck=strongcheck)
                _run(_worker, wslist, ncpu)
                
        drawimgkwargs.update({"rot_pair":False, "tru_type":drawcatkwargs["tru_type"]})
        wslist= makeworkerlist(workdir, catalogs, basename_list, drawimgkwargs, skipdone,  ext='_galimg.fits', strongcheck=strongcheck)
        _run(_worker, wslist, ncpu)

                        
        
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


def transformtogrid(inputdir, outputdir, drawcatkwargs):
        catalogs= sorted(glob.glob(os.path.join(inputdir, "*_cat.fits")))
        assert len(catalogs)>0
        outcatalogs=[os.path.join(outputdir, os.path.basename(cat).replace("_cat.fits","grid_cat.fits")) for cat in catalogs]
        for icat, ocat in zip(catalogs,outcatalogs):
                fromrandomtogrid(icat, drawcatkwargs, outputcatname=ocat)
                
        
def fromrandomtogrid(inputcat, drawcatkwargs, outputcatname=None):
        logger.info("Replacing in a grid %s"%(inputcat))
        incat_df=open_fits(inputcat)
        constcat_df=open_fits(inputcat,hdu=2)
        nimgs=constcat_df['nimgs'][0]

        ngal= drawcatkwargs['ngal']
        imagesize=drawcatkwargs['imagesize']
        assert ngal is not None
        assert imagesize is not None

        ntotalgal=len(incat_df)
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
                incat_df=open_fits(icat)
                constcat_df=open_fits(icat, hdu=2)
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
   
