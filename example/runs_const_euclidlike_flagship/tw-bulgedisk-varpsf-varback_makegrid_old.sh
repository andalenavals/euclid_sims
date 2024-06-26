#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims/
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/tw-bulgedisk-varpsf-varback-grid-big
BLENDEDDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/tw-bulgedisk-varpsf-varback-blended-vardensity/sim

SIMDIR=$WORKDIR/sim
SEXDIR=$WORKDIR/sex1.0
ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss_oldfunc
KSBDIR=$WORKDIR/ksb_trucat_nw
GROUPCATS=groupcat.fits
ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw.yaml
SEX_ARGS=$SCRIPTDIR/configfiles/sexconfigfiles/oldsexconf.yaml
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml
cd $GLOBAL../TF_MomentsML_experiments/examples/TF_fullimages_sims-meas/

python run_sim_meas_constimg_euclidlike.py --loglevel=INFO --transformtogriddir=$BLENDEDDIR --simdir=$SIMDIR --sexdir=$SEXDIR --adamomdir=$ADAMOMDIR --ksbdir=$KSBDIR --adamompsfcatalog=$ADAMOMPSFCAT --groupcats=$GROUPCATS --cat_args=$CAT_ARGS --sex_args=$SEX_ARGS --tru_type=2 --pixel_conv --dist_type=flagship --cattype=sex --ncpu=100 --ncat=2000 --typegroup=tw --skipdone --substractsky --runadamom --matchinput #--transformtogrid --runsims --usepsfimg --usevarpsf --usevarsky --runsex #--run_check #--use_weight 



