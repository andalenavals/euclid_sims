"""
Shape measurement with GalSim's adaptive moments (from its "hsm" module).
"""

import numpy as np
import sys, os
import fitsio
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import utils
import galsim


psf_oversampling = 5

def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def ksb_moments(stamp,xc=None,yc=None,sigw=2.0,prec=0.01):

        stamp_size=stamp.shape[0]
        
        # initialize the grid
        y,x=np.mgrid[:stamp_size,:stamp_size]
        
        dx=1.0
        dy=1.0

        if xc==None:
            xc=stamp_size*0.5
        if yc==None:
            yc=stamp_size*0.5

            
        # first recenter the weight function

        ntry=0
        while (abs(dx)>prec and abs(dy)>prec and ntry<10):
                
                w = gauss2d(x, y, xc, yc, sigw, sigw)
                ftot=np.sum(w*stamp)

                if ftot>0:

                        wx= x-xc
                        wy= y-yc
                       
                        dx,dy=np.sum(w*wx*stamp)/ftot, np.sum(w*wy*stamp)/ftot
                        xc=xc+dx
                        yc=yc+dy
                
                ntry=ntry+1
                        
        # compute the polarisation
        
        w = gauss2d(x, y, xc, yc, sigw, sigw)
        
        xx= np.power(x-xc,2)
        xy= np.multiply(x-xc,y-yc)
        yy= np.power(y-yc,2)
        
        q11=np.sum(xx*w*stamp)
        q12=np.sum(xy*w*stamp)
        q22=np.sum(yy*w*stamp)

        denom= q11 + q22

        if denom!=0.0:
                
                e1=(q11 - q22) / denom
                e2=(2. * q12) / denom

                # compute KSB polarisabilities
                # need to precompute some of this and repeat to speed up
        
                wp= -0.5 / sigw**2 * w
                wpp= 0.25 / sigw**4 * w        
        
                DD = xx + yy
                DD1 = xx - yy
                DD2 = 2. * xy

                Xsm11= np.sum((2. * w + 4. * wp * DD + 2. * wpp * DD1 * DD1) * stamp)
                Xsm22= np.sum((2. * w + 4. * wp * DD + 2. * wpp * DD2 * DD2) * stamp)
                Xsm12= np.sum((2. * wpp * DD1 * DD2) * stamp)
                Xsh11= np.sum((2. * w * DD + 2. * wp * DD1 * DD1) * stamp)
                Xsh22= np.sum((2. * w * DD + 2. * wp * DD2 * DD2) * stamp)
                Xsh12= np.sum((2. * wp * DD1 * DD2) * stamp)                

                em1 = np.sum((4. * wp + 2. * wpp * DD) * DD1 * stamp) / denom
                em2 = np.sum((4. * wp + 2. * wpp * DD) * DD2 * stamp) / denom               
                eh1 = np.sum(2. * wp * DD * DD1 * stamp) / denom + 2. * e1
                eh2 = np.sum(2. * wp * DD * DD2 * stamp) / denom + 2. * e2            
        
                psm11= Xsm11/ denom - e1 * em1
                psm22= Xsm11/ denom - e2 * em2
                psm12= Xsm12/ denom - 0.5 * (e1 * em2 + e2 * em1)              

                psh11= Xsh11 / denom - e1 * eh1
                psh22= Xsh22 / denom - e2 * eh2
                psh12= Xsh12 / denom - 0.5 * (e1 * eh2 + e2 * eh1)                

                ksbpar={}
                ksbpar['xc']=xc
                ksbpar['yc']=yc
                ksbpar['e1']=e1
                ksbpar['e2']=e2
                ksbpar['Psm11']=psm11
                ksbpar['Psm22']=psm22
                ksbpar['Psh11']=psh11
                ksbpar['Psh22']=psh22     
                
                return ksbpar
   
def measure_moments(shape_cat,gal_img,metacal_type,sigma_fix,psf_par):

        stepsize=0.02
        stepfac=1.+stepsize
        stampsize=20
        image_size=gal_img.shape[0]
    
        x_shift=[]; y_shift=[]
        e1_iso=[]; e2_iso=[]; 
        e1_ani=[]; e2_ani=[]
        flag_mom=[]
    
        x_gal=shape_cat['x_se'].astype(float) 
        y_gal=shape_cat['y_se'].astype(float)

        if (metacal_type=='noshear'):
                x_gal=x_gal-50
                y_gal=y_gal-50
                
        if (metacal_type=='1p'):
                
                x_gal=(x_gal-2000)*stepfac+1950
                y_gal=(y_gal-2000)/stepfac+1950
                
        elif (metacal_type=='1m'):
                
                x_gal=(x_gal-2000)/stepfac+1950
                y_gal=(y_gal-2000)*stepfac+1950
                
        elif (metacal_type=='2p'):
                
                x_gal=x_gal-2000
                y_gal=y_gal-2000
                xr=(x_gal/np.sqrt(2)-y_gal/np.sqrt(2))/stepfac
                yr=(x_gal/np.sqrt(2)+y_gal/np.sqrt(2))*stepfac
                x_gal=xr/np.sqrt(2)+yr/np.sqrt(2)+1950
                y_gal=-xr/np.sqrt(2)+yr/np.sqrt(2)+1950
                
        elif (metacal_type=='2m'):
                
                x_gal=x_gal-2000
                y_gal=y_gal-2000
                xr=(x_gal/np.sqrt(2)-y_gal/np.sqrt(2))*stepfac
                yr=(x_gal/np.sqrt(2)+y_gal/np.sqrt(2))/stepfac
                x_gal=xr/np.sqrt(2)+yr/np.sqrt(2)+1950
                y_gal=-xr/np.sqrt(2)+yr/np.sqrt(2)+1950        

        for i in range(0,len(x_gal)): 

                moments_ok=False
        
                xmin=np.int(x_gal[i])-stampsize
                xmax=np.int(x_gal[i])+stampsize+1
                ymin=np.int(y_gal[i])-stampsize
                ymax=np.int(y_gal[i])+stampsize+1        
        
                if (xmin>0 and xmax<image_size and ymin>0 and ymax<image_size):

                        xp=19.+x_gal[i]-np.int(x_gal[i])
                        yp=19.+y_gal[i]-np.int(y_gal[i]) 
                        stamp=gal_img[ymin:ymax,xmin:xmax]

                        try:
                                ksb_par=ksb_moments(stamp,xp,yp,sigw=sigma_fix)

                                dx=ksb_par['xc']-xp
                                dy=ksb_par['yc']-yp            

                                moments_ok=True
           
                                if (abs(dx)>1.5 or abs(dy)>1.5):
                    
                                        moments_ok=False

                        except:

                                moments_ok=False
       
                              
                if moments_ok==True:
            
                        x_shift.append(round(dx,3))
                        y_shift.append(round(dy,3))
                        e1_iso.append(round(ksb_par['e1'],4))
                        e2_iso.append(round(ksb_par['e2'],4))
                        e1_ani.append(round(ksb_par['Psm11'] * (psf_par['e1']/psf_par['Psm11']),4))
                        e2_ani.append(round(ksb_par['Psm22'] * (psf_par['e2']/psf_par['Psm22']),4))            
                        flag_mom.append(0)            
            
                else:
                        
                        x_shift.append(-99)
                        y_shift.append(-99)            
                        e1_iso.append(0.0)
                        e2_iso.append(0.0)
                        e1_ani.append(0.0)
                        e2_ani.append(0.0)
                        flag_mom.append(1)

        return e1_iso,e2_iso,e1_ani,e2_ani,x_shift,y_shift,flag_mom


def measure(imgfile, catalog, psf_obs, xname="x", yname="y", fwhm=None, weight=None,  filename=None,  skystats=True, nmaskedpix=True,aperture_r=20, extra_cols=None, use_weight=True):
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

        if len(catalog)==0:
                logger.info("Catalog %s have not galaxies. To noisy?"%(imgfile))
                return
        
        prefix="adamom_"
        if type(imgfile) is str:
                logger.debug("Loading FITS image %s..." % (imgfile))
                print("Loading FITS image %s..." % (imgfile))
                logger.info("Loading FITS image %s..." % (imgfile))
                img = galsim.fits.read(imgfile)
                img.setOrigin(0,0)
                logger.debug("Done with loading %s, shape is %s" % (imgfile, img.array.shape))
                imagesize=img.array.shape[0]

        if (weight is not None) & (type(weight) is str):
                logger.debug("Loading FITS image weight %s..." % (os.path.basename(weight)))
                segmap_img=weight
                data_seg=fitsio.read(segmap_img)
                img_seg=galsim.Image(data_seg, xmin=0,ymin=0)

       
        sigs_stamp=10
        stampsize=64
        # And loop
        data = []; data_sky=[]
        
        for i, gal in enumerate(catalog):

                flag=0
                        
                #logger.info("Using stampsize %i"%(stampsize))
                        
                (x, y) = (gal[xname], gal[yname])
                pos = galsim.PositionD(x,y)
                
                lowx=int(x-0.5*stampsize)+1
                lowy=int(y-0.5*stampsize)+1
                upperx=int(x+0.5*stampsize)
                uppery=int(y+0.5*stampsize)
                if lowx <0 :flag=BORDER_IMG ;lowx=0
                if lowy <0 :flag=BORDER_IMG ;lowy=0
                if upperx >= imagesize : flag=BORDER_IMG; upperx =imagesize-1
                if uppery >= imagesize : flag=BORDER_IMG; uppery =imagesize-1
                bounds = galsim.BoundsI(lowx,upperx , lowy , uppery ) # Default Galsim convention, index starts at 1.
                gps = img[bounds]
                if weight is not None:
                        img_seg_stamp = img_seg[bounds]
                        center = galsim.PositionI(int(x), int(y))
                        indx_use = [ 0, img_seg_stamp[center]]
                        mask = np.isin(img_seg_stamp.array,  indx_use)*1
                        gps_w = galsim.Image(mask, xmin=lowx,ymin=lowy)
                        if ~use_weight: gps_w=None

                        if nmaskedpix:
                                ny=uppery - lowy + 1
                                nx=upperx - lowx + 1
                                a, b = int(np.floor(y - lowy)), int(np.floor(x - lowx) )
                                r=aperture_r
                                yc,xc=np.ogrid[-a:ny-a, -b:nx-b]
                                mask_c= xc*xc+yc*yc<=r*r
                                tot_usedpix=np.sum(mask*mask_c )
                                maskedpixels=np.sum(mask_c)-tot_usedpix
                                fracpix=maskedpixels/tot_usedpix

                                X, Y = np.meshgrid(np.arange(nx) - b, np.arange(ny) - a)
                                distance_ar = np.sqrt(X**2 + Y**2)
                                Dinv = 1.0/np.clip( distance_ar, 1, aperture_r)

                                fracpix_md = np.sum( (~mask)*(mask_c)*Dinv )/ np.sum(mask_c*Dinv)
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
                        

                try:
                    if fwhm is None:
                        res = galsim.hsm.FindAdaptiveMom(gps,  weight=gps_w, guess_centroid=pos)
                        fwhm= 2.35482*res.moments_sigma
                        dx, dy, e1, e2, T, flux, flag = ngmix_fit(gps, gps_w, fwhm, x, y)
                    else:
                        dx, dy, e1, e2, T, flux, flag = ngmix_fit(gps, gps_w, fwhm, x, y)
                    if flag==BAD_MEASUREMENT: continue
                except:
                    logger.debug("NGMIX with default settings failed on:\n %s" % (str(gal)), exc_info = True)              
                    continue # skip to next stamp !
                
                

                ngmix_flux = res.moments_amp
                ngmix_x = res.moments_centroid.x + 1.0 # Not fully clear why this +1 is needed. Maybe it's the setOrigin(0, 0).
                ngmix_y = res.moments_centroid.y + 1.0 # But I would expect that GalSim would internally keep track of these origin issues.
                ngmix_g1 = res.observed_shape.g1
                ngmix_g2 = res.observed_shape.g2
                ngmix_sigma = res.moments_sigma
 
                
                if ngmix_flux<0: continue
                

                adamom_list=[ngmix_flux, ngmix_x, ngmix_y, ngmix_g1, ngmix_g2, ngmix_sigma]                        

                fields_names=[ "ngmix_%s"%(n) for n in  ["flux", "x", "y", "g1", "g2", "sigma"] ]
                if skystats:
                        out = utils.skystats(gps)
                        sky_list=[out["std"], out["mad"], out["mean"], out["med"],out["stampsum"]]
                        adamom_list+=sky_list
                        fields_names+=["skystd","skymad","skymean","skymed","skystampsum"]
                        
                if nmaskedpix:
                        neis_list=[maskedpixels, fracpix, fracpix_md, r_nmpix, galseg_area]
                        adamom_list+=neis_list
                        fields_names+=['mpix', 'fracpix', 'fracpix_md', 'r_nmpix', 'galseg_area']
                if extra_cols is not None:
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
