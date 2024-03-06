import glob
import logging
import sys
import os
import shlex
import fitsio
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

def run_with_timeout(cmd, timeout_sec):
    # cf. https://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
    import subprocess, shlex
    from threading import Timer
    from subprocess import DEVNULL

    subprocess.run(shlex.split(cmd), stdin=subprocess.PIPE, stdout=DEVNULL, stderr=DEVNULL)
    '''
    proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)
    proc.wait()
    
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout,stderr=proc.communicate()
    finally: 
        timer.cancel()
    
    stdout,stderr=proc.communicate()
    return stdout,stderr
    '''
def add_stream_handler(logger, level):
    formatter = logging.Formatter('%(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(level)  # Not sure if both this and sh.setLevel are required...


def run_SEXTRACTORPP(img_file, cat_file, sex_filter, sex_bin, sex_config, sex_params, python_config, psf_file,  check_flags, nthreads, logger):
    """Run sextractor, but only if the output file does not exist yet.??
    """
    logger.info('Running SourceSextractorPlusPlus')

    if python_config is not None:
        python_config = "--python-config-file %s"%(python_config)
        if psf_file is None:
            python_arg = "--python-arg=%s"%("image_file=%s"%(img_file))
        else:
            python_arg = "--python-arg=%s"%("image_file=%s psf_file=%s"%(img_file,psf_file))
    else:
        python_arg = ""

    if sex_filter is not None:
        s_filter='--segmentation-filter %s'%(sex_filter)
    else:
        s_filter= ""
    
    cat_cmd = "{sex_bin} --detection-image {img_file} --config-file {config} --output-catalog-filename {cat_file} {sex_params} {filter} {python_config} {python_arg} {check_flags} --thread-count {nthreads}".format( sex_bin=sex_bin, img_file=img_file, config=sex_config,  cat_file=cat_file,  filter=s_filter, check_flags=check_flags,  sex_params=sex_params,  python_config=python_config,  python_arg=python_arg, nthreads=nthreads)

    logger.info("FULL CMD COMMAND: %s"%(cat_cmd))
    run_with_timeout(cat_cmd, 120)

    '''
    stdout,stderr=run_with_timeout(cat_cmd, 120)
    if not os.path.exists(cat_file) or os.path.getsize(cat_file) == 0:
        logger.info('   Error running SExtractor++.  No ouput file was written.')
        logger.info('   Try again, in case it was a fluke.')
        stdout,stderr=run_with_timeout(cat_cmd, 120)
        if os.path.getsize(cat_file) == 0:
            os.remove(cat_file)
            logger.info('   Error running SExtractor++ (again).')
            return None
    '''
def read_sexparam(sexparam_path):
    out = []
    try:
        f = open(sexparam_path, 'r')
        lines = f.readlines()
    except OSError :
            with open(sexparam_path) as file: raise
            
    for line in lines:
        lex = shlex.shlex(line)
        lex.whitespace = '\n' # if you want to strip newlines, use '\n'
        line = ''.join(list(lex))
        if not line:
            continue
        out.append(line)
    
    return ",".join(out)

def get_checkflags(img_path):
    cflags = ["check-image-model-fitting", "check-image-residual", "check-image-background",
                  "check-image-variance", "check-image-segmentation", "check-image-partition",
                  "check-image-grouping", "check-image-filtered", "check-image-thresholded", 
                  "check-image-auto-aperture", "check-image-aperture", "check-image-psf"]
    check_dir = os.path.expanduser(os.path.join(img_path,"check"))
    try:
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)
    except OSError:
        if not os.path.exists(check_dir): raise
    check_flags_list = ["--%s %s"%(name, os.path.join(check_dir, "%s.fits"%(name) )) for name in cflags]
    check_flags = " ".join(check_flags_list)
    return check_flags

def run_SEGMAP( img_file, cat_file, check_file=None, sex_bin=None, sex_config=None, sex_params=None, sex_filter=None, sex_nnw=None):
    """Run sextractor, but only if the output file does not exist yet.
    """
    logger.info('running sextractor')
    if check_file is not None: check_flags="-CHECKIMAGE_NAME %s"%(check_file)
    else: check_flags=""
        
    cat_cmd = "{sex_bin} {img_file} -c {config} -CATALOG_NAME {cat_file} -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME {params} -FILTER_NAME {filter} -STARNNW_NAME {nnw} {check_flags} ".format(
        sex_bin=sex_bin, img_file=img_file, config=sex_config,
        cat_file=cat_file, params=sex_params, filter=sex_filter,
        nnw=sex_nnw ,check_flags=check_flags)
    


    logger.info("FULL CMD COMMAND: %s"%(cat_cmd))
    #stdout,stderr=run_with_timeout(cat_cmd, 120)
    run_with_timeout(cat_cmd, 120)
    
    '''
    print(stderr.decode())
    if len(stderr)>0:
        try:
            img=fitsio.read(img_file)
        except:
            logger.info("Un able to read %s. Removing it"%(img_file))
            os.remove(img_file)
            return None
        
    if not os.path.exists(cat_file) or os.path.getsize(cat_file) == 0:
        logger.info('   Error running SExtractor.  No ouput file was written.')
        logger.info('   Try again, in case it was a fluke.')
        stdout,stderr=run_with_timeout(cat_cmd, 120)
        if os.path.getsize(cat_file) == 0:
            os.remove(cat_file)
            logger.info('   Error running SExtractor (again).')
            return None
    '''
    

def add_ellipsepars(catalog):
    output = fitsio.read(catalog)
    output= output.astype(output.dtype.newbyteorder('='))
    output = pd.DataFrame(output)
    if "AWIN_IMAGE" in output.columns:
        output["ELONGATION_WIN"]=output["AWIN_IMAGE"]/output["BWIN_IMAGE"]
        output["ELLIP_WIN"]=(output["ELONGATION_WIN"]-1)/(output["ELONGATION_WIN"]+1) # always > 0
        output["ELLIP1_WIN"]=output["ELLIP_WIN"]*np.cos(np.deg2rad( 2*output["THETAWIN_IMAGE"]))
        output["ELLIP2_WIN"]=output["ELLIP_WIN"]*np.sin(np.deg2rad( 2*output["THETAWIN_IMAGE"]))
        output["ELLIP_AREA"]=np.pi*output["AWIN_IMAGE"]*output["BWIN_IMAGE"]
        fitsio.write(catalog, output.to_records(index=False), clobber=True)
    else: return
