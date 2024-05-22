import numpy as np
import pandas as pd
import random
import astropy.io.fits as fits
import copy
import yaml

def tru_sersicn_func(tru_sersicn_tmp):
        tru_sersicns = np.linspace(0.3, 6.0, 21)
        tru_sersicn = tru_sersicns[(np.abs(tru_sersicns-tru_sersicn_tmp)).argmin()]
        return tru_sersicn
def trunc_rayleigh(sigma, max_val):
        """
        A truncated Rayleigh distribution
        """
        assert max_val > sigma
        tmp = max_val + 1.0
        while tmp > max_val:
                tmp = np.random.rayleigh(sigma)
        return tmp
          

def draw_position(imagesize, idx=None, ngals=None, mode='grid'):   
    if (mode=='grid'):
        aux=np.sqrt(ngals)
        if int(aux)==aux:
                nparts_side= aux
        else:
                nparts_side= int(aux) +1
            
        (piy, pix) = divmod(idx, nparts_side)
        stampsize=imagesize/nparts_side
        x=(pix+0.5)*stampsize
        y=(piy+0.5)*stampsize
    else: # mode random default
        x=imagesize*np.random.uniform(0.0, 1.0)
        y=imagesize*np.random.uniform(0.0, 1.0)
    return x, y
#just more efficient draw to avoid looping
def draw_position_sample(nsamples, imagesize, ngals=None, mode='grid'):
        if (mode=='grid'):
                aux=np.sqrt(ngals)
                if int(aux)==aux:
                        nparts_side= aux
                else:
                        nparts_side= int(aux) +1

                idxs=[ i for i in range(ngals)]
                (piy, pix) = np.vectorize(divmod)(idxs, nparts_side)
                stampsize=imagesize/nparts_side
                x=(pix+0.5)*stampsize
                y=(piy+0.5)*stampsize
                assert nsamples%ngals==0
                nimgs=int(nsamples/ngals)
                x=x.tolist()*nimgs
                y=y.tolist()*nimgs
                
        else: # mode random default
                x=(imagesize*np.random.uniform(0.0, 1.0,size=nsamples)).tolist()
                y=(imagesize*np.random.uniform(0.0, 1.0,size=nsamples)).tolist()
        return x, y
def get_cat(cat, rmin, rmax, magmin, magmax,sernmin,sernmax, dr,dmag,dserc):
        r_u = np.random.uniform(rmin, rmax) 
        mag_u = np.random.uniform(magmin, magmax)
        sersicn_u = np.random.uniform(sernmin, sernmax)
        flagr= (cat["bulge_r50"]< r_u+0.5*dr)&(cat["bulge_r50"]>r_u-0.5*dr)
        flagmag= (cat["tru_mag"]< mag_u+0.5*dmag)&(cat["tru_mag"]>mag_u-0.5*dmag)
        flagsersic= (cat["bulge_nsersic"]< sersicn_u+0.5*dserc)&(cat["bulge_nsersic"]>sersicn_u-0.5*dserc)
        aux=cat[flagr&flagmag&flagsersic]
        return aux
def get_cat_nolims(cat, nbins=25):
        rmin=np.min(cat["bulge_r50"]); rmax=np.max(cat["bulge_r50"]); dr=(rmax-rmin)/nbins
        magmin=np.min(cat["tru_mag"]); magmax=np.max(cat["tru_mag"]); dmag=(magmax-magmin)/nbins
        sernmin=np.min(cat["bulge_nsersic"]); sernmax=np.max(cat["bulge_nsersic"]); dserc=(sernmax-sernmin)/nbins
        r_u = np.random.uniform(rmin, rmax) 
        mag_u = np.random.uniform(magmin, magmax)
        sersicn_u = np.random.uniform(sernmin, sernmax)
        flagr= (cat["bulge_r50"]< r_u+0.5*dr)&(cat["bulge_r50"]>r_u-0.5*dr)
        flagmag= (cat["tru_mag"]< mag_u+0.5*dmag)&(cat["tru_mag"]>mag_u-0.5*dmag)
        flagsersic= (cat["bulge_nsersic"]< sersicn_u+0.5*dserc)&(cat["bulge_nsersic"]>sersicn_u-0.5*dserc)
        aux=cat[flagr&flagmag&flagsersic]
        return aux
        
def uniflagshipdraw(ind, sourcecat=None, constants=None):
        print("Drawing uniflagship", ind)
        exptime=constants["exptime"]
        gain=constants["gain"]
        zeropoint=constants["zeropoint"]
        with fits.open(sourcecat) as hdul:
                cat=hdul[1].data
                hdul.close()
                
        leng=0
        while leng==0:
                auxcat=get_cat_nolims(copy.deepcopy(cat), nbins=25)
                leng=len(auxcat)
                        
        row = random.choice(auxcat)
        tru_mag = row["tru_mag"]
        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
        tru_bulge_g=float(row["bulge_ellipticity"])
        tru_theta=row["disk_angle"]
        (tru_bulge_g1, tru_bulge_g2) = (tru_bulge_g*np.cos(2.0 * tru_theta*(np.pi/180.)), tru_bulge_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
        tru_bulge_flux=float(row["bulge_fraction"]*tru_flux)
        tru_bulge_sersicn_tmp=float(row["bulge_nsersic"])
        tru_bulge_rad=float(row["bulge_r50"])
        tru_disk_rad=float(row["disk_r50"])
        tru_disk_flux=float((1-row["bulge_fraction"])*tru_flux)
        tru_disk_inclination=float(row["inclination_angle"])#* galsim.degrees
        tru_disk_scaleheight=float(row["disk_scalelength"])
        dominant_shape=row["dominant_shape"]
        bulge_axis_ratio=row["bulge_axis_ratio"]
        #cosmos_index=row["cosmos_index"]
        return  tru_mag, tru_flux, tru_bulge_g, tru_theta, tru_bulge_g1, tru_bulge_g2, tru_bulge_flux, tru_bulge_sersicn_tmp, tru_bulge_rad, tru_disk_rad, tru_disk_flux,tru_disk_inclination, tru_disk_scaleheight , dominant_shape, bulge_axis_ratio#, cosmos_index

def draw(tru_type=1, dist_type="gems",  sourcecat=None, constants=None):

        exptime=constants["exptime"]
        gain=constants["gain"]
        zeropoint=constants["zeropoint"]
        if(tru_type==1): #if Sersic
                tru_g = trunc_rayleigh(0.25, 0.7)
                tru_theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0)                
                (tru_g1, tru_g2) = (tru_g * np.cos(2.0 * tru_theta), tru_g * np.sin(2.0 * tru_theta))

                if (dist_type == "gems"):
                        #sourcecat = fitsio.read(gemssourcecat)
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                        source_row = random.choice(cat)

                        tru_rad = source_row["tru_rad"]
                        tru_mag = source_row["tru_mag"]
                
                        # The sersicn is "tmp", as we discretize it further below
                        tru_sersicn_tmp = source_row["tru_sersicn"]
                elif (dist_type == "uni"):
                        """
                        Uniform only distributions 
                        """
                        tru_rad = np.random.uniform(0.1, 1.0) #in arcsec
                        tru_mag = np.random.uniform(20.5, 25.0)
                        tru_sersicn_tmp = np.random.uniform(0.3, 6.0)
                        
                elif (dist_type == "uniflagship"):
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                                hdul.close()
                                cat=cat[cat["dominant_shape"]==0] #only bulge
                        nbins=25
                        rmin=0.1; rmax=1.0 # in arcsec
                        magmin=20.5; magmax=25.0
                        sernmin=0.3; sernmax=6.0
                        dr=(rmax-rmin)/nbins
                        dmag=(magmax-magmin)/nbins
                        dserc=(sernmax-sernmin)/nbins

                        leng=0
                        while leng==0:
                                auxcat=get_cat(copy.deepcopy(cat), rmin, rmax, magmin, magmax,sernmin,sernmax, dr,dmag,dserc)
                                leng=len(auxcat)
                                
                        row = random.choice(auxcat)
                        tru_g=float(row["bulge_ellipticity"])
                        tru_theta=row["disk_angle"]
                        (tru_g1, tru_g2) = (tru_g * np.cos(2.0 * tru_theta*(np.pi/180.)), tru_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
                        tru_mag = float(row["tru_mag"])
                        tru_sersicn_tmp=float(row["bulge_nsersic"])
                        tru_rad=float(row["bulge_r50"])         

                elif (dist_type == "flagship"):
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                                hdul.close()
                                cat=cat[cat["dominant_shape"]==0] #only bulge
                        row = random.choice(cat)
                        tru_g=float(row["bulge_ellipticity"])
                        tru_theta=row["disk_angle"]
                        (tru_g1, tru_g2) = (tru_g * np.cos(2.0 * tru_theta*(np.pi/180.)), tru_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
                        tru_mag = float(row["tru_mag"])
                        tru_sersicn_tmp=float(row["bulge_nsersic"])
                        tru_rad=float(row["bulge_r50"])
      

                tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
                tru_sersicn=tru_sersicn_func(tru_sersicn_tmp)

                out = {"tru_flux":tru_flux,
                       "tru_rad":tru_rad,
                       "tru_g1":tru_g1,
                       "tru_g2":tru_g2,
                       "tru_sersicn":tru_sersicn,
                       "tru_g":tru_g,
                       "tru_theta":tru_theta,
                       "tru_mag": tru_mag}

        elif(tru_type==2): #if BulgeDisk1
                if (dist_type == "uni"):
                        """
                        Uniform only distributions -- can this work ?
                        """
                        tru_disk_rad = np.random.uniform(0.1, 1.0) # in arcsec    
                        tru_bulge_rad = np.random.uniform(0.1, 1.0) # in arcsec
                        tru_mag = np.random.uniform(20.5, 25.0)
                        #tru_mag = np.random.uniform(20.0, 26.0)
                        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint)) #in ADU
                        tru_bulge_g = trunc_rayleigh(0.25, 0.7)
                        tru_theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0)   

                        tru_bulge_sersicn_tmp = np.random.uniform(0.5, 6.0)

                elif (dist_type == "uniflagship"):
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                                hdul.close()

                        leng=0
                        while leng==0:
                                auxcat=get_cat_nolims(copy.deepcopy(cat), nbins=25)
                                leng=len(auxcat)
                        
                        row = random.choice(auxcat)
                        tru_mag = row["tru_mag"]
                        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
                        tru_bulge_g=float(row["bulge_ellipticity"])
                        tru_theta=row["disk_angle"]
                        (tru_bulge_g1, tru_bulge_g2) = (tru_bulge_g*np.cos(2.0 * tru_theta*(np.pi/180.)), tru_bulge_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
                        tru_bulge_flux=float(row["bulge_fraction"]*tru_flux)
                        tru_bulge_sersicn_tmp=float(row["bulge_nsersic"])
                        tru_bulge_rad=float(row["bulge_r50"])
                        tru_disk_rad=float(row["disk_r50"])
                        tru_disk_flux=float((1-row["bulge_fraction"])*tru_flux)
                        tru_disk_inclination=float(row["inclination_angle"])#* galsim.degrees
                        tru_disk_scaleheight=float(row["disk_scalelength"])
                        dominant_shape=row["dominant_shape"]
                        bulge_axis_ratio=float(row["bulge_axis_ratio"])


                elif(dist_type == "flagship"):
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                                hdul.close()
                        row = random.choice(cat)
                        tru_mag = row["tru_mag"]
                        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
                        tru_bulge_g=float(row["bulge_ellipticity"])
                        tru_theta=row["disk_angle"]
                        (tru_bulge_g1, tru_bulge_g2) = (tru_bulge_g*np.cos(2.0 * tru_theta*(np.pi/180.)), tru_bulge_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
                        tru_bulge_flux=float(row["bulge_fraction"]*tru_flux)
                        tru_bulge_sersicn_tmp=float(row["bulge_nsersic"])
                        tru_bulge_rad=float(row["bulge_r50"])
                        tru_disk_rad=float(row["disk_r50"])
                        tru_disk_flux=float((1-row["bulge_fraction"])*tru_flux)
                        tru_disk_inclination=float(row["inclination_angle"])#* galsim.degrees
                        tru_disk_scaleheight=float(row["disk_scalelength"])
                        dominant_shape=row["dominant_shape"]
                        bulge_axis_ratio=float(row["bulge_axis_ratio"])

                

                tru_bulge_sersicn = tru_sersicn_func(tru_bulge_sersicn_tmp) 

    
                out = {"tru_mag": tru_mag,
                       "tru_flux":tru_flux,
                       "bulge_ellipticity":tru_bulge_g,
                       "tru_theta":tru_theta,
                       "disk_angle":tru_theta,
                       "tru_g1":tru_bulge_g1,
                       "tru_g2":tru_bulge_g2,
                       "tru_bulge_flux":tru_bulge_flux,
                       "bulge_nsersic":tru_bulge_sersicn,
                       "bulge_r50":tru_bulge_rad,
                       "disk_r50":tru_disk_rad,
                       "tru_disk_flux":tru_disk_flux,
                       "inclination_angle": tru_disk_inclination,
                       "disk_scalelength":tru_disk_scaleheight,
                       "dominant_shape":dominant_shape,
                       "bulge_axis_ratio":bulge_axis_ratio,
                       #"cosmos_index":row["cosmos_index"],
                }
        return out

#juxst more efficient draw to avoid looping
def draw_sample(nsamples,tru_type=1, dist_type="gems",  sourcecat=None, constants=None):
        exptime=constants["exptime"]
        gain=constants["gain"]
        zeropoint=constants["zeropoint"]

        if(tru_type==1): #if Sersic
                tru_g = np.vectorize( trunc_rayleigh)([0.25]*nsamples, 0.7)
                tru_theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0, size=nsamples)                
                (tru_g1, tru_g2) = (tru_g * np.cos(2.0 * tru_theta), tru_g * np.sin(2.0 * tru_theta))

                if (dist_type == "gems"):
                        """
                        Takes values from a random GEMS galaxy
                        """
                        #cat = fitsio.read(sourcecat)
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                        rows = np.random.choice(cat, size=nsamples)

                        tru_rad = rows["tru_rad"] 
                        tru_mag = rows["tru_mag"] 
                        
                        # The sersicn is "tmp", as we discretize it further below
                        tru_sersicn_tmp = rows["tru_sersicn"]
                elif (dist_type == "uni"):
                        """
                        Uniform only distributions -- can this work ?
                        """
                        tru_rad = np.random.uniform(0.1, 1.0, size=nsamples)
                        tru_mag = np.random.uniform(20.5, 25.0, size=nsamples)
                        tru_sersicn_tmp = np.random.uniform(0.3, 6.0, size=nsamples)
                        
                elif (dist_type == "flagship"):
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                                hdul.close()
                                cat=cat[cat["dominant_shape"]==0] #only bulge
                
                        rows = np.random.choice(cat, size=nsamples)
                        tru_g=rows["bulge_ellipticity"]
                        tru_theta=rows["disk_angle"]
                        (tru_g1, tru_g2) = (tru_g * np.cos(2.0 * tru_theta*(np.pi/180.)), tru_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
                        tru_mag = rows["tru_mag"]
                        tru_sersicn_tmp=rows["bulge_nsersic"]
                        tru_rad=rows["bulge_r50"]
                        bulge_axis_ratio=rows["bulge_axis_ratio"]


                tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
                tru_sersicn = np.vectorize(tru_sersicn_func)(tru_sersicn_tmp)
                out = {"tru_flux":tru_flux,
                       "tru_rad":tru_rad,
                       "tru_g1":tru_g1,
                       "tru_g2":tru_g2,
                       "tru_sersicn":tru_sersicn,
                       "tru_g":tru_g,
                       "tru_theta":tru_theta,
                       "bulge_axis_ratio":bulge_axis_ratio,
                       "tru_mag": tru_mag,}

        elif(tru_type==2): #if BulgeDisk1
                if (dist_type == "uni"):
                        """
                        Uniform only distributions TODO
                        """
          
                        tru_mag = np.random.uniform(20.5, 25.0, size=nsamples)
                        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
                        tru_bulge_g = np.vectorize( trunc_rayleigh)([0.25]*nsamples, 0.7)
                        tru_theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0, size=nsamples)   

                        tru_sersicn_tmp = np.random.uniform(0.5, 6.0, size=nsamples)
                elif (dist_type == "uniflagship"):
                       tru_mag, tru_flux, tru_bulge_g, tru_theta, tru_bulge_g1, tru_bulge_g2, tru_bulge_flux, tru_bulge_sersicn_tmp, tru_bulge_rad, tru_disk_rad, tru_disk_flux,tru_disk_inclination, tru_disk_scaleheight , dominant_shape, bulge_axis_ratio = np.vectorize(uniflagshipdraw)(np.arange(nsamples), sourcecat=sourcecat, constants=constants)

                elif(dist_type == "flagship"):
                        with fits.open(sourcecat) as hdul:
                                cat=hdul[1].data
                                hdul.close()
                        row = np.random.choice(cat, size=nsamples)
                        tru_mag = row["tru_mag"]
                        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
                        tru_bulge_g=row["bulge_ellipticity"]
                        tru_theta=row["disk_angle"]
                        (tru_bulge_g1, tru_bulge_g2) = (tru_bulge_g*np.cos(2.0 * tru_theta*(np.pi/180.)), tru_bulge_g * np.sin(2.0 * tru_theta*(np.pi/180.)))
                        tru_bulge_flux=row["bulge_fraction"]*tru_flux
                        tru_bulge_sersicn_tmp=row["bulge_nsersic"]
                        tru_bulge_rad=row["bulge_r50"]
                        tru_disk_rad=row["disk_r50"]
                        tru_disk_flux=(1-row["bulge_fraction"])*tru_flux
                        tru_disk_inclination=row["inclination_angle"] #* galsim.degrees
                        tru_disk_scaleheight=row["disk_scalelength"]
                        dominant_shape=row["dominant_shape"]
                        bulge_axis_ratio=row["bulge_axis_ratio"]

                tru_bulge_sersicn = np.vectorize(tru_sersicn_func)(tru_bulge_sersicn_tmp)
                #tru_sersicn = tru_bulge_sersicn_tmp
    
                out = {"tru_mag": tru_mag,
                       "tru_flux":tru_flux,
                       "bulge_ellipticity":tru_bulge_g,
                       "tru_theta":tru_theta,
                       "disk_angle":tru_theta,
                       "tru_g1":tru_bulge_g1,
                       "tru_g2":tru_bulge_g2,
                       "tru_bulge_flux":tru_bulge_flux,
                       "bulge_nsersic":tru_bulge_sersicn,
                       "bulge_r50":tru_bulge_rad,
                       "disk_r50":tru_disk_rad,
                       "tru_disk_flux":tru_disk_flux,
                       "inclination_angle": tru_disk_inclination,
                       "disk_scalelength":tru_disk_scaleheight,
                       "dominant_shape":dominant_shape,
                       "bulge_axis_ratio":bulge_axis_ratio,
                       #"cosmos_index":row["cosmos_index"],
                }
        return out

def draw_sample_star(nsamples, sourcecat=None, constants=None):
        exptime=constants["exptime"]
        gain=constants["gain"]
        zeropoint=constants["zeropoint"]
        with fits.open(sourcecat) as hdul:
                cat=hdul[1].data
                hdul.close()
        row = np.random.choice(cat, size=nsamples)
        tru_mag = row["tru_mag"]
        tru_flux =  (exptime / gain) * 10**(-0.4*(tru_mag - zeropoint))
        out={"tru_flux":tru_flux, "tru_mag":tru_mag} 
        return out

def draw_s(max_shear):
    """
    Shear and magnification
    """

    #(row["gamma1"], row["gamma2"])
    
    if max_shear > 0.0 and max_shear < 10.0:
            tru_s1 = np.random.uniform(-max_shear, max_shear)
            tru_s2 = np.random.uniform(-max_shear, max_shear)                
    else:
            tru_s1 = 0.0
            tru_s2 = 0.0
        
    tru_mu = 1.0
    return {"tru_s1": tru_s1, "tru_s2": tru_s2, "tru_mu": tru_mu}

def draw_ngal(nmin=1, nmax=20, nbins=10):
        #ngal = random.randint(nmin, nmax)
        binsize=int((nmax-nmin)/nbins)+1
        ngal=np.random.choice([nmin + i*binsize for i in range(nbins)])
        return ngal

