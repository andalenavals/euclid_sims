#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/tp-bulgedisk-varpsf-varback-grid-samereas_nsub2_biguesig_flagship_fixtheta


ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml


cd $SCRIPTDIR

python make_uniformtp.py --loglevel=INFO --workdir=$WORKDIR  --constants=$CONSTANTS --tru_type=2 --dist_type=flagship --ncpu=100 --ncat=10000 --skipdone --npairs=2048 --patience=25 --subsample_nbins=2 --runadamom --drawcat --usevarpsf --usevarsky 
