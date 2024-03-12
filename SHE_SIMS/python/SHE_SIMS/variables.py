import numpy as np

GALCOLS=[ "tru_rad", "tru_sb", "tru_sersicn", "tru_flux","tru_g", "tru_mag"]+["tru_bulge_flux","tru_disk_flux", "tru_bulge_rad","tru_disk_rad", "tru_disk_inclination", "tru_bulge_sersicn", "tru_disk_scaleheight", "dominant_shape"]+["disk_angle","disk_scalelength","bulge_r50","disk_r50", "inclination_angle", "bulge_ellipticity", "bulge_nsersic", "tru_bulge_g", "bulge_axis_ratio"]
STARCOLS=["star_flag","r_star", "star_flux", "star_mag"]

SEXFEATS =["X_IMAGE", "Y_IMAGE", "MAG_AUTO", "SNR_WIN", "FLAGS", "FWHM_IMAGE", "FLUX_RADIUS","FLUX_WIN", "ELONGATION_WIN", "ELLIP_WIN", "ELLIP1_WIN", "ELLIP2_WIN", "ELLIP_AREA", "FLUXERR_WIN", "MAGERR_AUTO", "MAG_WIN",  "MAG_PSF", "PETRO_RADIUS", "KRON_RADIUS",  "SPREADERR_MODEL", "CLASS_STAR", ] +["ELONGATION","AWIN_IMAGE", "BWIN_IMAGE" , "THETAWIN_IMAGE", "ERRAWIN_IMAGE", "ERRBWIN_IMAGE", "ERRTHETAWIN_IMAGE" , "CXXWIN_IMAGE", "CYYWIN_IMAGE", "CXYWIN_IMAGE", "ERRCXXWIN_IMAGE", "ERRCYYWIN_IMAGE", "ERRCXYWIN_IMAGE" ]

PSFCOLS =  [ "%s%s"%("psf_adamom_", n) for n in ["flux", "g1", "g2", "sigma", "rho4"]]+["quadrant", "ccd","z", "e1", "e2", "R2", "sed"]+[ "%s%s"%("psf_mom_", n) for n in ["flux", "g1", "g2", "sigma", "rho4", "M4_1","M4_2"]]

MEASCOLS=["adamom_%s"%(f) for f in ["flux","g1", "g2", "sigma", "rho4", "x", "y"]]+["centroid_shift"]+['gal_density']+["skymad","snr"]+['psf_g1_ksb', 'psf_sigma_ksb', 'psf_g2_ksb']+["corr_s1","corr_s2"]+["ngmix_dx", "ngmix_dy","ngmix_e1", "ngmix_e2", "ngmix_T", "ngmix_flux" ]+np.concatenate([[ "ngmix_%s_%s%s"%("moments",f,"corr"),"ngmix_%s_%s%s"%("moments",f,""), "ngmix_%s_%s%s"%("fit",f,"corr")] for f in ["flags","T","T_err","s2n","g1","g2","g1_err","g2_err"]]).tolist()+np.concatenate([[ "ngmix_%s_%s%s"%("moments",f,"corr"),"ngmix_%s_%s%s"%("moments",f,"")] for f in ["flux","flux_err"]]).tolist()
       
MATCHCOLS=["tru_rad_match", "tru_mag_match", "tru_r_n1_match", "r_match"]

NEISCOLS=np.concatenate([["fracpix%i"%(i),"fracpix_md%i"%(i)] for i in [10,15,20,25,30]]).tolist() +["mpix",  "fracpix", "fracpix_md","r_nmpix", 'galseg_area']
neicols_tru=["tru_r_n1", "tru_r_n2"]+["tru_flux_n1","tru_rad_n1", "tru_mag_n1"]+["tru_flux_n2","tru_rad_n2", "tru_mag_n2"]
neicols_sex=["SEX_R_n1", "SEX_R_n2", "SNR_WIN_n1", "SNR_WIN_n2", "MAG_AUTO_n1", "MAG_AUTO_n2"]
neicols_ada=["adamom_r_n1", "adamom_flux_n1", "adamom_sigma_n1"]+["adamom_r_n2", "adamom_flux_n2", "adamom_sigma_n2"]

COLS1D=["tru_s1", "tru_s2", "tru_gal_density", "tru_sky_level"]
COLS2D=['tru_g1', "tru_g2", "x", "y", "tru_theta"]+["tru_bulge_g1", "tru_bulge_g2"]
