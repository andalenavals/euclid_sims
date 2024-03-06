"""
Shape measurement with GalSim's adaptive moments (from its "hsm" module).
"""
import scipy
from scipy.linalg import fractional_matrix_power
import numpy as np
import sys, os
import fitsio
import pandas as pd
from astropy.io import fits
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import utils
from .. import calc
import galsim


profiles=["Gaussian", "Sersic", "EBulgeDisk"]
RLIST=[10,15,20,25,30]

def measure(imgfile, catalog,  xname="x", yname="y", variant="default", weight=None,  filename=None,  skystats=True, nmaskedpix=True,aperture_r=20, extra_cols=None, use_weight=True, rot_pair=False, skipdone=False, substractsky=True, edgewidth=5):
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
                logger.debug("Adding %i extra cols in galsim_adamom measure"%(len(extra_cols)))
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
        
        prefix="adamom_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                #print("Loading FITS image %s..." % (imgfile))
                logger.info("Loading FITS image %s..." % (imgfile))
                try:
                        img = galsim.fits.read(imgfile)
                        #img.setOrigin(0,0)
                        #galaxys were draw with the galsim and sextractor convention (1,1) 
                except:
                        logger.info("exception trying to read image %s"%(imgfile))
                        if os.path.exists(imgfile):
                                logger.info("removing corrupted image")
                                #os.remove(imgfile)
                                return None
                logger.debug("Done with loading %s, shape is %s" % (imgfile, img.array.shape))
                imagesize=img.array.shape[0]

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
        
        for i, gal in enumerate(catalog):

                flag=0
                        
                #logger.info("Using stampsize %i"%(stampsize))
                        
                (x, y) = (gal[xname], gal[yname]) # I set origin to (0,0)
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
                        
                        indx_use = [ 0, img_seg_stamp[center]]
                        mask = np.isin(img_seg_stamp.array,  indx_use) #True means use, False reject
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
                                                r_nmpix = r

                                galseg = np.isin(img_seg_stamp.array,  [img_seg_stamp[center]] )*1
                                galseg_area=np.sum(galseg)
                        
                else:
                        gps_w = None
                        nmaskedpix=False
                        assert ~use_weight
                        
                # And now we measure the moments... galsim may fail from time to time, hence the try:
                get_moments(gps, gps_w=None)

                ada_flux = res.moments_amp
                ada_x = res.moments_centroid.x + 1.0 # Not fully clear why this +1 is needed. Maybe it's the setOrigin(0, 0).
                ada_y = res.moments_centroid.y + 1.0 # But I would expect that GalSim would internally keep track of these origin issues.
                ada_g1 = res.observed_shape.g1
                ada_g2 = res.observed_shape.g2
                ada_sigma = res.moments_sigma
                ada_rho4 = res.moments_rho4
                
                if ada_flux<0: continue
                

                adamom_list=[ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_sigma, ada_rho4]                        

                fields_names=[ "adamom_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma", "rho4"] ]
                if skystats:
                        sky_list=[sky["std"], sky["mad"], sky["mean"], sky["med"],sky["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, r_nmpix, galseg_area]+fracpix+ fracpix_md,
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


        endtime = datetime.now()        
        logger.info("All done")

        n = len(catalog)
        nfailed = n - len(data)
        
        logger.info("I failed on %i out of %i sources (%.1f percent)" % (nfailed, n, 100.0*float(nfailed)/float(n)))
        logger.info("This measurement took %.3f ms per galaxy" % (1e3*(endtime - starttime).total_seconds() / float(n)))



def Adap_w(x, y, x0, y0, M11, M12, M22):
        xs1 = x - x0  
        xs2 = y - y0  
        M = [[M11, M12],[M12, M22]]
        M = np.linalg.inv(M)
        return np.exp(-1/2*((M[0,0]*xs1+M[0,1]*xs2)*xs1+ (M[1,0]*xs1 + M[1,1]*xs2)*xs2)) 

def E(pars, xs1, xs2, stamp):
        A, x01, x02, M11, M12, M22 = pars
        xs1 = xs1 - x01
        xs2 = xs2 - x02
        M = [[M11, M12],[M12, M22]]
        M = np.linalg.inv(M)
        hh = np.zeros((stamp.shape[0], stamp.shape[1]))
        hh = (M[0,0]*xs1+M[0,1]*xs2)*xs1+ (M[1,0]*xs1 + M[1,1]*xs2)*xs2
        return 1/2 * np.sum(np.square(stamp - A * np.exp(-1/2*np.array(hh))  ))
        
def get_moments(gps, gps_w=None):
        '''
        gps: image or array with the target object in the center
        gps_w:  image or array with the weight to mask regions of the gps image
        '''
        if isinstance(gps, galsim.Image):
                stamp=gps.array
        if isinstance(gps_w, galsim.Image):
                stamp_w=gps_w.array
        
        stamp_size=stamp.shape[0]
        y,x=np.mgrid[:stamp_size,:stamp_size]
        
        initial_guess=[0.01,stamp_size*0.5,stamp_size*0.5,15,1,15]
        res = scipy.optimize.minimize(E,x0 = initial_guess, args= (x, y, stamp))
        A, x01, x02, M11, M12, M22 = res.x
        xx= np.power(x-x01,2)
        xy= np.multiply(x-x01,y-x02)
        yy= np.power(y-x02,2)

        w_adap =  Adap_w(x, y, x01, x02, M11, M12, M22)
        q11_adap=np.sum(xx*w_adap*stamp)/np.sum(w_adap*stamp)
        q12_adap=np.sum(xy*w_adap*stamp)/np.sum(w_adap*stamp)
        q22_adap=np.sum(yy*w_adap*stamp)/np.sum(w_adap*stamp)

        T=q11_adap + q22_adap
        e1_adap = (q11_adap - q22_adap)/T
        e2_adap = (2*q12_adap)/T
        

        
        M = [[q11_adap, q12_adap],[q12_adap, q22_adap]]
        M_inv_half_adap = fractional_matrix_power(M, (-1/2))
        u_adap = M_inv_half_adap[0][0]*(x-x01)+M_inv_half_adap[0][1]*(y-x02)
        v_adap = M_inv_half_adap[1][0]*(x-x01)+M_inv_half_adap[1][1]*(y-x02)
        # print(u_adap)
        # print(x-x01)
        
        u4_adap = np.power(u_adap,4)
        v4_adap = np.power(v_adap,4)
        u3v_adap = np.multiply(np.power(u_adap,3),v_adap)
        v3u_adap = np.multiply(np.power(v_adap,3),u_adap)
        u2v2_adap = np.multiply(np.power(u_adap,2),np.power(v_adap,2))
        
        M40_adap = np.sum(u4_adap*w_adap*stamp)/np.sum(w_adap*stamp)
        M04_adap = np.sum(v4_adap*w_adap*stamp)/np.sum(w_adap*stamp)
        M31_adap = np.sum(u3v_adap*w_adap*stamp)/np.sum(w_adap*stamp) 
        M13_adap = np.sum(v3u_adap*w_adap*stamp)/np.sum(w_adap*stamp)
        M22_adap = np.sum(u2v2_adap*w_adap*stamp)/np.sum(u2v2_adap*stamp)
        
        M4_1_adap = M40_adap - M04_adap
        M4_2_adap = 2*M31_adap + 2*M13_adap
        xc = x01
        yc = x02
                

        ada_flux = A
        ada_x = xc 
        ada_y = yc
        ada_g1 = e1_adap
        ada_g2 = e2_adap
        ada_size = T
        ada_sigma = np.power(q11_adap*q22_adap-q12_adap**2,1/4)
        ada_M4_1 = M4_1_adap
        ada_M4_2 = M4_2_adap
        ada_rho4 = M40_adap + M04_adap + 2*M22_adap
        return [ada_flux, ada_x, ada_y, ada_g1, ada_g2, ada_size, ada_sigma, ada_M4_1, ada_M4_2,ada_rho4]  
