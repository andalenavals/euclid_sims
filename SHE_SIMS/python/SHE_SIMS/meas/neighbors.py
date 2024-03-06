"""
This measfct adds a SNR to the catalog, computed from existing columns.
It does not measure the images.

"""
import os, glob

import fitsio
from astropy.io import fits
import pandas as pd
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import copy
import datetime
import multiprocessing

import logging
logger = logging.getLogger(__name__)

def measfct_old(catalog, cols, n_neis=1, prefix ='adamom_'):
        
        """
        catalog
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        n_neis: include cols upto this number of neighbors (sorted by distance)
        
        """
        logger.info("Starting neighbors features measurements")
        bad_flags=[-999,-888]
                
        output = fitsio.read(catalog)
        output= output.astype(output.dtype.newbyteorder('='))
        output = pd.DataFrame(output)
        nimg_case=max(output["img_id"])+1
        output["global_img_id"]= output["cat_id"]*nimg_case +output["img_id"]

        meascat_dfs=[]
        for g_img_id in set(output["global_img_id"]):
                logger.info("Adding neighbors to image %i"%(g_img_id))
                meascat_df=output[output["global_img_id"]==g_img_id].reset_index(drop=True)
                cat_pos= meascat_df[["%sx"%(prefix),"%sy"%(prefix)]]
                mask=~np.any(cat_pos.isin(bad_flags),axis=1).to_numpy()
                #cat_pos.mask(cat_pos.isin(bad_flags),inplace=True)
                cat_pos.mask = cat_pos.isin(bad_flags)
                dists=cdist(cat_pos,cat_pos)
                sort_idxs=np.argsort(dists, axis=1) #get indeces that sort dists
                for i in range(n_neis):
                        r=np.partition(dists, i+1,axis=1)[:,i+1]
                        meascat_df['r_n%i'%(i+1)]=r
                        meascat_df.loc[~mask,'r_n%i'%(i+1)]=-999
                
                        idxs = sort_idxs.T[i+1] #i+1 closest neighbor index (row) in meascat
                        idxs = idxs[mask]
                        for col in cols:
                                meascat_df["%s_n%i"%(col,i+1)] = -999
                                meascat_df.loc[mask,"%s_n%i"%(col,i+1)]=meascat_df.loc[idxs,col].to_numpy()
        
                meascat_dfs.append(meascat_df)
        imgsmeascats=pd.concat(meascat_dfs, ignore_index=True)

        fitsio.write(catalog, imgsmeascats.to_records(index=False), clobber=True)


#ADD NEIGHBOR FEATURES TO MEASURED CATALOG
def measfct(measdir, cols=[], ext='_meascat.fits' ,n_neis=1, xname ='X_IMAGE', yname='Y_IMAGE', r_label='r', hdu=1, skipdone=False, ncpu=1):
        
        """
        Measure DETECTED neighbors
        measdir: directory with measured properties (can be sex or meas)
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        n_neis: include cols upto this number of neighbors (sorted by distance)
        
        """
        logger.info("Starting neighbors features measurements")

        input_catalogs=sorted(glob.glob(os.path.join(measdir,'*_img','*%s'%(ext) )) )

        wslist=[]
        for catname in input_catalogs:
                logger.info("Doing cat %s"%(catname))
                
                wc=_NEISSettings(catname, cols, n_neis, xname,yname, r_label, hdu, skipdone)
                wslist.append(wc)

        _neisrun(wslist, ncpu)
     
class _NEISSettings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, catname, cols, n_neis, xname,yname, r_label, hdu, skipdone):
                self.catname = catname
                self.cols = cols
                self.n_neis = n_neis
                self.xname = xname
                self.yname = yname
                self.r_label = r_label
                self.hdu = hdu
                self.skipdone=skipdone
                
def _neisworker(ws):
        """
        Worker function that the different processes will execute, processing the _NEISSettings
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        p = multiprocessing.current_process()
        logger.info("%s is starting neibors measure catalog %s with PID %s" % (p.name, str(ws.catname), p.pid))

        cat = fitsio.read(ws.catname, ext=ws.hdu)
        cat= cat.astype(cat.dtype.newbyteorder('='))
        cat_df = pd.DataFrame(cat)

        if len(cat_df)<= ws.n_neis:
                logger.info("Warning your detections are smaller than the requested number of neighbors skipping cat")
                return

        ws.cols=[ col for col in ws.cols if col in cat_df.columns]
        
        newcols=[]
        for col in ws.cols:
                newcols+=['%s_n%i'%(col,i) for i in range(1, ws.n_neis+1)]

        if ws.skipdone:
                li=len(set(cat_df.columns.to_numpy()).intersection(set(newcols)))
                if li==len(newcols):
                        logger.info("Measurement file exist and have all existing extra column")
                        return
                
        cat_pos= cat_df[[ws.xname,ws.yname]]
        tree=scipy.spatial.KDTree(cat_pos)
        distin,indin=tree.query(cat_pos,k=ws.n_neis+1)
        cat_df[["%s_n%i"%(ws.r_label,i) for i in range(1, ws.n_neis+1) ]]=distin[:,1:]

        for col in ws.cols:
                cat_df[['%s_n%i'%(col,i) for i in range(1, ws.n_neis+1)]]= cat_df[col].to_numpy()[indin[:,1:]]
                
        catbintable = fits.BinTableHDU(cat_df.to_records(index=False))
        with fits.open(ws.catname) as hdul:
                hdul.pop(ws.hdu)
                hdul.insert(ws.hdu, catbintable)
                hdul.writeto(ws.catname, overwrite=True)
       
        
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))

def _neisrun(wslist, ncpu):
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
        
        logger.info("Getting neigbors features on %i images using %i CPUs" % (len(wslist), ncpu))
        
        if ncpu == 1: # The single process way (MUCH MUCH EASIER TO DEBUG...)
                list(map(_neisworker, wslist))
        
        else:
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(_neisworker, wslist)
                pool.close()
                pool.join()
        
        endtime = datetime.datetime.now()
        logger.info("Done, the neigbors features measurement time was %s" % (str(endtime - starttime)))




#ADD NEIGHBOR FEATURES TO INPUT CATALOG   
def measfct_trucat(simdir, cols=[], ext='_cat.fits' ,n_neis=1, xname ='x', yname='y', r_label='tru_r', hdu=1, skipdone=False, ncpu=1):
        
        """
        Measure DETECTED neighbors
        measdir: directory with measured properties (can be sex or meas)
        cols: measured cols to add to the catalog, not need to specify distance it will be default
        n_neis: include cols upto this number of neighbors (sorted by distance)
        
        """
        logger.info("Starting neighbors features measurements")

        input_catalogs=sorted(glob.glob(os.path.join(simdir,'*%s'%(ext) )) )
        wslist=[]
        for catname in input_catalogs:
                logger.info("Doing cat %s"%(catname))
                wc=_NEISTRUSettings(catname, cols, n_neis, xname,yname, r_label, hdu, skipdone)
                wslist.append(wc)
        _neistrurun(wslist, ncpu)
                

class _NEISTRUSettings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, catname, cols, n_neis, xname,yname, r_label, hdu, skipdone):
                self.catname = catname
                self.cols = cols
                self.n_neis = n_neis
                self.xname = xname
                self.yname = yname
                self.r_label = r_label
                self.hdu = hdu
                self.skipdone=skipdone
                
def _neistruworker(ws):
        """
        Worker function that the different processes will execute, processing the _NEISSettings
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        p = multiprocessing.current_process()
        logger.info("%s is starting neibors measure catalog %s with PID %s" % (p.name, str(ws.catname), p.pid))
        cat = fitsio.read(ws.catname, ext=ws.hdu)
        cat= cat.astype(cat.dtype.newbyteorder('='))
        cat_df = pd.DataFrame(cat)

        ws.cols=[ col for col in ws.cols if col in cat_df.columns]
        
        newcols=[]
        for col in ws.cols:
                newcols+=['%s_n%i'%(col,i) for i in range(1, ws.n_neis+1)]

        if ws.skipdone:
                inter=set(cat_df.columns.to_numpy()).intersection(set(newcols))
                li=len(inter)
                if li==len(newcols):
                        logger.info("Measurement file exist and have all existing extra column")
                        return
                else:
                        logger.info("Adding %s"%(str(inter)))
                
        nimgs=max(cat["img_id"])+1
        for img_id in range(nimgs):
                logger.info("Doing img_id %i"%(img_id))
                cataux=cat_df[cat_df['img_id']==img_id]
                
                cat_pos= cataux[[ws.xname,ws.yname]]
                tree=scipy.spatial.KDTree(cat_pos)
                distin,indin=tree.query(cat_pos,k=ws.n_neis+1)
                
                cat_df.loc[(cat_df['img_id']==img_id) ,["%s_n%i"%(ws.r_label,i) for i in range(1, ws.n_neis+1) ]]=distin[:,1:]
                
                for col in ws.cols:
                        cat_df.loc[ (cat_df['img_id']==img_id) ,['%s_n%i'%(col,i) for i in range(1, ws.n_neis+1)]]= cat_df.loc[(cat_df['img_id']==img_id), col].to_numpy()[indin[:,1:]]

        catbintable = fits.BinTableHDU(cat_df.to_records(index=False))
        with fits.open(ws.catname) as hdul:
                hdul.pop(1)
                hdul.insert(1, catbintable)
                hdul.writeto(ws.catname, overwrite=True)
       
        
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))

def _neistrurun(wslist, ncpu):
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
        
        logger.info("Getting neigbors features on %i images using %i CPUs" % (len(wslist), ncpu))
        
        if ncpu == 1: # The single process way (MUCH MUCH EASIER TO DEBUG...)
                list(map(_neistruworker, wslist))
        
        else:
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(_neistruworker, wslist)
                pool.close()
                pool.join()
        
        endtime = datetime.datetime.now()
        logger.info("Done, the total neigbors features measurement time was %s" % (str(endtime - starttime)))




#ADD TRUE ICS TO INPUT CATALOGS
def add_trueics(catsdir, ext='_cat.fits'):
        logger.info("Adding ics to input catalogs")
        input_catalogs=sorted(glob.glob(os.path.join(catsdir, '*%s'%(ext)) ))
        
        rmax=20
        nneis=5
        faint_mag=25.0
        
        for catname in input_catalogs:
                logger.info("Doing cat %s"%(catname))
                cat = fitsio.read(catname)
                cat= cat.astype(cat.dtype.newbyteorder('='))
                cat = pd.DataFrame(cat)

                if 'ics' in cat.columns.to_numpy(): continue
                nimgs=max(cat["img_id"])+1
                for img_id in range(nimgs):
                        logger.info("Doing img_id %i"%(img_id))
                        cataux=cat[cat['img_id']==img_id]
                        cat_pos=cataux[['x','y']]
                        tree=scipy.spatial.KDTree(cat_pos)
                        distin,indin=tree.query(cat_pos,k=nneis+1)

                        mags=cat['tru_mag'].to_numpy()[indin]
                        diff_mag=faint_mag-mags
                        diff_mag[diff_mag<0]=0
                        diff_mag[distin>rmax]=0
                        #removing first self galaxy, and sum
                        ics=np.sum(diff_mag[:,1:],axis=1)

                        
                        cat.loc[(cat['img_id']==img_id), 'ics']=ics

                catbintable = fits.BinTableHDU(cat.to_records(index=False))#name='true_variable_properties'
                with fits.open(catname) as hdul:
                        hdul.pop(1)
                        hdul.insert(1, catbintable)
                        hdul.writeto(catname, overwrite=True)



#ADD ICS TO DETECTION CATALOGS
def add_ics(detectdir, inputdir, xname='X_IMAGE', yname='Y_IMAGE', ext='_cat.fits', ncpu=1, skipdone=False):
        logger.info("Adding ics to detection catalogs")
        detectdirs=sorted(glob.glob(os.path.join(detectdir,'*_img') ))

        wslist=[]
        for ddir in detectdirs:
                basename=os.path.basename(ddir).replace("_img", "")
                inputcatname=os.path.join(inputdir, '%s%s'%(basename, '_cat.fits'))
                incat=fitsio.read(inputcatname)
                incat= incat.astype(incat.dtype.newbyteorder('='))
                incat = pd.DataFrame(incat)
                nimgs=max(incat["img_id"])+1
                for img_id in range(nimgs):
                        detectcatname=os.path.join(ddir,'%s_img%i_galimg%s'%(basename, img_id ,'%s'%(ext)))
                        wc=_ICSWorkerSettings(incat,detectcatname,img_id,skipdone)
                        wslist.append(wc)
        _icsrun(wslist, ncpu)

class _ICSWorkerSettings():
        """
        A class that holds together all the settings for running sextractor in an image.
        """
        
        def __init__(self, incat, detectcatname, img_id, skipdone):
                self.incat= incat
                self.detectcatname= detectcatname
                self.img_id= img_id
                self.skipdone= skipdone
                
def _icsworker(ws):
        """
        Worker function that the different processes will execute, processing the
        _SexWorkerSettings objects.
        """
        starttime = datetime.datetime.now()
        np.random.seed()
        p = multiprocessing.current_process()
        logger.info("%s is starting Sextractor measure catalog %s with PID %s" % (p.name, str(ws), p.pid))

        incat,detectcatname, img_id=ws.incat, ws.detectcatname, ws.img_id
        xname='X_IMAGE'; yname='Y_IMAGE'
        
        rmax=20
        nneis=4

        
        try:
                with fits.open(detectcatname) as hdul:
                        hdul.verify('fix')
                        dcat=hdul[2].data
                        dcat= dcat.astype(dcat.dtype.newbyteorder('='))
                        dcat = pd.DataFrame(dcat)
        except Exception as e:
                logger.info("Failed reading sextractor catalog %s removing it"%(detectcatname))
                logger.info(str(e))
                if os.path.isfile(detectcatname):
                        os.remove(detectcatname)
                return
                                
                        

        if ws.skipdone:
                if 'ICS2' in dcat.columns.to_numpy():
                        logger.info("Skiping %s already in catalogs"%(detectcatname))
                        return

        logger.info("Doing %s"%(detectcatname))
       
        cataux=incat[incat['img_id']==img_id]
        itree=scipy.spatial.KDTree(cataux[['x','y']])
        distin,indin=itree.query(dcat[[xname,yname]],k=nneis+1)

        dist_mask=(distin>rmax)


        #ics=distin[:,1:]
        
        
        '''
        mags=cataux['tru_mag'].to_numpy()[indin]
        mags[distin>rmax]=0
        ics=np.sum(mags[:,1:],axis=1)#/mags[:,0]
        ics=np.sum(distin[:,1:]*mags[:,1:],axis=1)/np.clip(np.sum(distin[:,1:],axis=1),1,np.inf)
        '''

                        
        #without considering matched galaxy properties
        
        mags=cataux['tru_mag'].to_numpy()[indin]
        faint_mag=26.0
        #faint_mag=mags[:,0].reshape(mags.shape[0],1)
        diff_mag=faint_mag-mags
        
        diff_mag_mask=(diff_mag<0)
        mask=diff_mag_mask|dist_mask
        diff_mag[mask]=0
        ics=np.sum(diff_mag[:,1:],axis=1)
        #ics=np.sum((1.0/distin[:,1:])*diff_mag[:,1:],axis=1)/np.clip(np.sum((1/distin[:,1:]),axis=1),1,np.inf)
        
        

        '''
        flux=cataux['tru_flux'].to_numpy()[indin]
        mask=dist_mask
        flux[mask]=0
        ratio=np.sum(flux[:,1:],axis=1)/flux[:,0]
        #ratio=np.sum(distin[:,1:]*flux[:,1:],axis=1)/(np.clip(np.sum(distin[:,1:],axis=1),1,np.inf)*flux[:,0])
        ratio[np.isclose(ratio,0)]=1
        ics=np.log10(ratio)
        '''
        

        '''
        rad=(cataux['tru_rad']**2).to_numpy()[indin]
        mask=dist_mask
        rad[mask]=0
        ratio=np.sum(flux[:,1:],axis=1)/flux[:,0]
        #ratio=np.sum(distin[:,1:]*flux[:,1:],axis=1)/(np.clip(np.sum(distin[:,1:],axis=1),1,np.inf)*flux[:,0])
        ratio[np.isclose(ratio,0)]=1
        ics=np.log10(ratio)
        '''
        
        
                
        dcat['ICS2']=ics

        catbintable = fits.BinTableHDU(dcat.to_records(index=False))
        with fits.open(detectcatname) as hdul:
                hdul.pop(2)
                hdul.insert(2, catbintable)
                hdul.writeto(detectcatname, overwrite=True)

        
        endtime = datetime.datetime.now()
        logger.info("%s is done, it took %s" % (p.name, str(endtime - starttime)))

def _icsrun(wslist, ncpu):
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
        
        logger.info("Getting ICS on %i images using %i CPUs" % (len(wslist), ncpu))
        
        if ncpu == 1: # The single process way (MUCH MUCH EASIER TO DEBUG...)
                list(map(_icsworker, wslist))
        
        else:
                pool = multiprocessing.Pool(processes=ncpu)
                pool.map(_icsworker, wslist)
                pool.close()
                pool.join()
        
        endtime = datetime.datetime.now()
        logger.info("Done, the total ICS measurement time was %s" % (str(endtime - starttime)))



'''
def add_ics(detectdir, inputdir, xname='X_IMAGE', yname='Y_IMAGE', ext='_cat.fits'):
        logger.info("Adding ics to detection catalogs")
        detectdirs=sorted(glob.glob(os.path.join(detectdir,'*_img') ))
         
        rmax=50
        nneis=4
        

        for ddir in detectdirs:
                basename=os.path.basename(ddir).replace("_img", "")
                inputcatname=os.path.join(inputdir, '%s%s'%(basename, '_cat.fits'))
                incat=fitsio.read(inputcatname)
                incat= incat.astype(incat.dtype.newbyteorder('='))
                incat = pd.DataFrame(incat)
                nimgs=max(incat["img_id"])+1
                for img_id in range(nimgs):
                        detectcatname=os.path.join(ddir,'%s_img%i_galimg%s'%(basename, img_id ,'%s'%(ext)))
                        try:
                                #dcat = fitsio.read(detectcatname, ext=2)
                                #dcat= dcat.astype(dcat.dtype.newbyteorder('='))
                                #dcat = pd.DataFrame(dcat)
                                with fits.open(detectcatname) as hdul:
                                        hdul.verify('fix')
                                        dcat=hdul[2].data
                                        dcat= dcat.astype(dcat.dtype.newbyteorder('='))
                                        dcat = pd.DataFrame(dcat)
                        except Exception as e:
                                logger.info("Failed reading sextractor catalog %s removing it"%(detectcatname))
                                logger.info(str(e))
                                if os.path.isfile(detectcatname):
                                        os.remove(detectcatname)
                                continue
                                
                        

                        #if 'ICS2' in dcat.columns.to_numpy():
                        #        logger.info("Skiping %s already in catalogs"%(detectcatname))
                        #        continue

                        logger.info("Doing %s"%(detectcatname))
                        
                        cataux=incat[incat['img_id']==img_id]
                        itree=scipy.spatial.KDTree(cataux[['x','y']])
                        distin,indin=itree.query(dcat[[xname,yname]],k=nneis+1)
                        dist_mask=(distin>rmax)
                        distin[dist_mask]=0
                        
                        mags=cataux['tru_mag'].to_numpy()[indin]
                        #mags[distin>rmax]=0
                        #ics=np.sum(mags[:,1:],axis=1)#/mags[:,0]
                        #ics=np.sum(distin[:,1:]*mags[:,1:],axis=1)/np.clip(np.sum(distin[:,1:],axis=1),1,np.inf)

                        
                        #without considering matched galaxy properties
                        faint_mag=26.0
                        diff_mag=faint_mag-mags
                        diff_mag_mask=(diff_mag<0)
                        mask=diff_mag_mask&dist_mask
                        diff_mag[mask]=0
                        ics=np.sum(diff_mag[:,1:],axis=1)

                        distin[mask]=0
                        #ics=np.sum(distin[:,1:]*diff_mag[:,1:],axis=1)/np.clip(np.sum(distin[:,1:],axis=1),1,np.inf)
                        
                        #removing first self galaxy, and sum
                        
                        
                        dcat['ICS2']=ics

                        catbintable = fits.BinTableHDU(dcat.to_records(index=False))
                        with fits.open(detectcatname) as hdul:
                                hdul.pop(2)
                                hdul.insert(2, catbintable)
                                hdul.writeto(detectcatname, overwrite=True)

'''
