#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/tw-bulgedisk-varpsf-varback-blended-vardensity
SIMDIR=$WORKDIR/sim
SEXDIR=$WORKDIR/sex1.0
ADAMOMDIR=$WORKDIR/adamom_sexcat_w_ss1.0
KSBDIR=$WORKDIR/ksb_sexcat_w_ss1.0
GROUPCATS=groupcat.fits
PSFFILESDIR=/vol/euclidraid4/data/aanavarroa/catalogs/PSFs_PSFToolkit2022_shiftUm2.0_big/
ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
#CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw-blended-vardensity.yaml
CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw-blended-vardensity_stars.yaml
SEX_ARGS=$SCRIPTDIR/configfiles/sexconfigfiles/oldsexconf.yaml
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml

cd $SCRIPTDIR

python run_sim_meas_constimg_euclidlike.py --loglevel=INFO --simdir=$SIMDIR --sexdir=$SEXDIR --adamomdir=$ADAMOMDIR  --ksbdir=$KSBDIR --psffilesdir=$PSFFILESDIR --adamompsfcatalog=$ADAMOMPSFCAT --groupcats=$GROUPCATS --cat_args=$CAT_ARGS --sex_args=$SEX_ARGS --constants=$CONSTANTS --tru_type=2 --pixel_conv --substractsky --dist_type=flagship --cattype=sex --ncpu=100 --ncat=1 --typegroup=tw --skipdone --runadamom --use_weight --nsubcases=0 --add_ics --runsex --run_check --runsims --usepsfimg --usevarsky --usevarpsf #--run_check  #--runsex --run_check --substractsky 



