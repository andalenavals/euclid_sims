#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/vo-bulgedisk-varpsf-varback-grid-big
BLENDEDDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/vo-bulgedisk-varpsf-varback-blended-vardensity/sim

SIMDIR=$WORKDIR/sim
SEXDIR=$WORKDIR/sex1.0
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss
#ADAMOMDIR=$WORKDIR/adamom_trucat_nw_ss
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss_sub2
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss_sub2_biguesig
#ADAMOMDIR=$WORKDIR/adamom_trucat_nw_ss_sub2_biguesig_stamp
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss_sub2_biguesig_fixestamp
ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss_sub2_biguesig_fixestamp_photutils
#ADAMOMDIR=$WORKDIR/adamom_trucat_nw_ss_sub2
KSBDIR=$WORKDIR/ksb_trucat_nw
GROUPCATS=groupcat.fits
ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw.yaml
SEX_ARGS=$SCRIPTDIR/configfiles/sexconfigfiles/oldsexconf.yaml
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml
cd $SCRIPTDIR

python run_sim_meas_constimg_euclidlike.py --loglevel=NULL --transformtogriddir=$BLENDEDDIR --simdir=$SIMDIR --sexdir=$SEXDIR --adamomdir=$ADAMOMDIR --ksbdir=$KSBDIR --adamompsfcatalog=$ADAMOMPSFCAT --groupcats=$GROUPCATS --cat_args=$CAT_ARGS --sex_args=$SEX_ARGS --constants=$CONSTANTS --tru_type=2 --pixel_conv --dist_type=flagship --cattype=sex --ncpu=40 --ncat=2000 --typegroup=tw --skipdone --substractsky --transformtogrid --subsample_nbins=2 --runadamom #--rot_pair #--matchinput --match_pairs  #--rot_pair #--runsims --usepsfimg --usevarpsf --usevarsky --runsex #--run_check #--use_weight 



