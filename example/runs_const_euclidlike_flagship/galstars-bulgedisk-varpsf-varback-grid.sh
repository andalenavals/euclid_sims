#!/bin/bash

GLOBAL=/users/aanavarroa/original_gitrepos/euclid_sims
SCRIPTDIR=$GLOBAL/example/
WORKDIR=/vol/euclidraid5/data/aanavarroa/catalogs/MomentsML2/fullimages_constimg_euclid-flagship0.05/galstars-bulgedisk-varpsf-varback-grid2
SIMDIR=$WORKDIR/sim
SEXDIR=$WORKDIR/sex1.0
#ADAMOMDIR=$WORKDIR/adamom_sexcat_nw_ss1.0
#ADAMOMDIR=$WORKDIR/adamom_sexcat_w_ss1.0
ADAMOMDIR=$WORKDIR/adamom_trucat_nw_ss_sub2

KSBDIR=$WORKDIR/ksb_sexcat_nw_ss1.0
GROUPCATS=groupcat.fits
ADAMOMPSFCAT=/vol/euclidraid4/data/aanavarroa/catalogs/all_adamom_PSFToolkit_2022_shiftUm2.0_big.fits
#CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw.yaml
CAT_ARGS=$SCRIPTDIR/configfiles/simconfigfiles/tw-stars.yaml
SEX_ARGS=$SCRIPTDIR/configfiles/sexconfigfiles/oldsexconf.yaml
CONSTANTS=$SCRIPTDIR/configfiles/simconstants.yaml
cd $SCRIPTDIR

python run_sim_meas_constimg_euclidlike.py --loglevel=INFO --simdir=$SIMDIR --sexdir=$SEXDIR --adamomdir=$ADAMOMDIR --ksbdir=$KSBDIR --adamompsfcatalog=$ADAMOMPSFCAT --groupcats=$GROUPCATS --cat_args=$CAT_ARGS --sex_args=$SEX_ARGS --constants=$CONSTANTS --tru_type=2 --pixel_conv --dist_type=flagship --cattype=tru --ncpu=200 --ncat=2000 --typegroup=tw --skipdone --substractsky --subsample_nbins=2 --runadamom --stars #--runsims --usepsfimg --usevarpsf --usevarsky --stars #--substractsky #--use_weight --matchinput #--runsex --run_check #--runsims --usepsfimg --usevarpsf --usevarsky --stars #--rot_pair --stars #--transformtogrid --runsims --usepsfimg #--usevarpsf --usevarsky --use_weight --runsex --run_check --runksb



