import argparse
import galsim
import os

def parse_args(): 
    parser = argparse.ArgumentParser(description='Basic code to produce simulations PSF cores')
    parser.add_argument('--sigma', default=1, type=float, 
                        help='PSF FWHM in pixes')
    parser.add_argument('--imagesize', default=63, type=int, 
                        help='Number of catalog to draw')
    parser.add_argument('--workdir',
                        default='/users/aanavarroa/original_gitrepos/TF_MomentsML_experiments/examples/TF_fullimages_sims-meas/configfiles/sexconfigs_psfimg/psf_img_production', 
                        help='diractory of work')
    parser.add_argument('--filename',
                        default='sig1_63pix.fits', 
                        help='diractory of work')
    args = parser.parse_args()
    return args

def make_dir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except OSError:
        if not os.path.exists(dirname): raise

def main():
    args = parse_args()

    make_dir(args.workdir)
    
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    image = galsim.ImageF(args.imagesize , args.imagesize)
    image.scale = 1.0

    psf = galsim.Gaussian(flux=1., sigma=args.sigma)        
    #psf = psf.shear(g1=row["tru_psf_g1"], g2=row["tru_psf_g2"])
    #ud = galsim.UniformDeviate()
    #psf_xjitter = ud() - 0.5
    #psf_yjitter = ud() - 0.5
    #psf = psf.shift(psf_xjitter,psf_yjitter)

    psf.drawImage(image, method="auto", add_to_image=True)

    filename=os.path.join(args.workdir, args.filename)
    image.write(filename)

if __name__ == "__main__":
    main()
    
