#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/vo-bulgedisk-varpsf-varback-blended-vardensity
SIMDIR=$WORKDIR/sim
SEXDIR=$WORKDIR/sex1.0
#ADAMOMDIR=$WORKDIR/adamom_sexcat_w_ss1.0_sub2
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss1.0_sub2
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss1.0_sub2_biguesig
ADAMOMDIR=$WORKDIR/adamom_sexcat_w_ss1.0_sub2_biguesig
#ADAMOMDIR=$WORKDIR/adamom_sexcat_w_ss1.0_sub2_biguesig_fixstamp
KSBDIR=$WORKDIR/ksb_sexcat_w_ss1.0
GROUPCATS=groupcat.fits
ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw-blended-vardensity.yaml
SEX_ARGS=$SCRIPTDIR/configfiles/sexconfigfiles/oldsexconf.yaml
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml

cd $SCRIPTDIR

python run_sim_meas_constimg_euclidlike.py --loglevel=INFO --simdir=$SIMDIR --sexdir=$SEXDIR --adamomdir=$ADAMOMDIR  --ksbdir=$KSBDIR --adamompsfcatalog=$ADAMOMPSFCAT --groupcats=$GROUPCATS --cat_args=$CAT_ARGS --sex_args=$SEX_ARGS --constants=$CONSTANTS --tru_type=2 --pixel_conv --substractsky --subsample_nbins=2 --dist_type=flagship --cattype=sex --ncpu=20 --ncat=2000 --typegroup=tw --skipdone --runadamom --rot_pair --use_weight #--matchinput --add_ics #--nsubcases=2 --subsample_nbins=2 #--rot_pair #--match_pairs --matchinput --use_weight #--add_ics --matchinput --match_pairs #--rot_pair --use_weight #--runsex --run_check --substractsky --runsims --usepsfimg --usevarsky --usevarpsf 



