"""
Not used in principle this is only for getting fracpix_md, after measurements.
which is now done during the measurement process
"""

import numpy as np
import pandas as pd
import sys, os, glob
import fitsio
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import utils
import galsim


profiles=["Gaussian", "Sersic", "EBulgeDisk"]

#adding columns with segmap info to measdir
#measdir should have been produced with sextractor positions
def add_segmap_features(measdir, sexdir, anuli=30):
        sexcatext="_cat.fits"
        checkext='_seg.fits'
        if sexdir is None:
                raise
        logger.info("Exploring the contents of measdir '%s'" % (measdir))
        casefolders = sorted(glob.glob(os.path.join(measdir, "*_img")))
        for folder in casefolders:
                meascats=sorted(glob.glob(os.path.join(folder, "*_meascat.fits")))
                for meascat in meascats:
                        base = os.path.splitext(meascat)[0].replace(measdir, sexdir).replace("_meascat","")
                        sexcat_file = '%s%s'%(base,sexcatext)
                        check_file = '%s%s'%(base,checkext)
                        addfeatures( meascat, sexcat_file, check_file, anuli=anuli)
                        

def addfeatures( meascat_file, sexcat_file, segmap_file,  xname="X_IMAGE", yname="Y_IMAGE", stampsize= 64 ,   anuli=30):
        """
        nmaskedpix: flag to include number of masked pixels
        stampsize: optimal stampsize. It must be > 2*anuli
        anuli: radi around xname and yname to check the number of masked pixels
        """
        starttime = datetime.now()
        
        sexcat=fitsio.read(sexcat_file,ext=2)
        meascat=fitsio.read(meascat_file,ext=1)
        meascat=meascat.astype(meascat.dtype.newbyteorder('='))
        meascat_df=pd.DataFrame(meascat)

        logger.info("Loading FITS image %s..." % (os.path.basename(segmap_file)))
        data_seg=fitsio.read(segmap_file)
        img_seg=galsim.Image(data_seg, xmin=0,ymin=0)

        imagesize = data_seg.shape[0]

        gal_id=[]; mpix=[];fpix=[]
        for gal in sexcat:
                (x, y) = (gal[xname], gal[yname])
                lowx=int(x-0.5*stampsize)+1
                lowy=int(y-0.5*stampsize)+1
                upperx=int(x+0.5*stampsize)
                uppery=int(y+0.5*stampsize)
                if lowx <0 : lowx=0
                if lowy <0 : lowy=0
                if upperx >= imagesize : upperx =imagesize-1
                if uppery >= imagesize : uppery =imagesize-1
                bounds = galsim.BoundsI(lowx,upperx , lowy , uppery )

                img_seg_stamp = img_seg[bounds]
                center = galsim.PositionI(int(x), int(y))
                indx_use = [ 0, img_seg_stamp[center]]
                mask = np.isin(img_seg_stamp.array,  indx_use)*1 #expensive
                #img_wgt=galsim.Image(mask, xmin=0,ymin=0)

                #not optimal in memory use
                #mask = np.isin(data_seg, [0, data_seg[a,b] ])*1 
                #img_wgt=galsim.Image(mask, xmin=0,ymin=0)
                
                ny=uppery - lowy + 1
                nx=upperx - lowx + 1
                a, b = int(np.floor(y - lowy)), int(np.floor(x - lowx) )
                r=anuli
                yc,xc=np.ogrid[-a:ny-a, -b:nx-b]
                mask_c= xc*xc+yc*yc<=r*r ##expensive               

                
                #a, b = int(y), int(x)
                #n = data_seg.shape[0]
                #yc,xc=np.ogrid[-a:n-a, -b:n-b]
                #mask_c= xc*xc+yc*yc<=r*r ##expensive
                #img_c =galsim.Image(mask_c*1, xmin=0,ymin=0)
                #img_c_stamp = img_c[bounds]
                #tot_usedpix=np.sum(img_wgt.array*img_c_stamp.array)
                #maskedpixels=np.sum(img_c_stamp.array)-tot_usedpix

                tot_usedpix=np.sum(mask*mask_c )
                maskedpixels=np.sum(mask_c)-tot_usedpix
                fracpix=maskedpixels/tot_usedpix

                gal_id.append(gal["NUMBER"]-1)
                mpix.append(maskedpixels)
                fpix.append(fracpix)

        df=pd.DataFrame({"obj_number":gal_id, "mpix":mpix, "fracpix":fpix})
        meascat_df = meascat_df.merge(df, on=["obj_number"])
        print(meascat_df)
                        
        fitsio.write(meascat_file, meascat_df.to_records(index=False), clobber=True)
        logger.info("measurements catalog %s written"%(meascat_file))


        endtime = datetime.now()        
        logger.info("All done, segmap features added")

     

