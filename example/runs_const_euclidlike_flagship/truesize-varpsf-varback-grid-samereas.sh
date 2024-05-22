#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/truesize-varpsf-varback-grid-samereas_nsub2

ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml

cd $SCRIPTDIR

python make_uniformtp.py --loglevel=INFO --workdir=$WORKDIR  --constants=$CONSTANTS --tru_type=2 --dist_type=uniflagship --ncpu=200 --ncat=5000 --skipdone --max_shear=0 #--runadamom --drawcat --usevarpsf --usevarsky --skipdone --npairs=50 --patience=100 --subsample_nbins=2
