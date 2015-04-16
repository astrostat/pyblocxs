from ui import *
import sherpa.astro.ui as ui
from mhtest import *
execfile("funcs.py")
import numpy as np

#dataspace1d(0.0,1024.0,id=1,dstype=DataPHA) 

load_pha("PCA/src_01_refake_a=2,nH=1e23,1e5.fak")
#load_pha("PCA/src_01_refake_a=1,nH=1e21,1e5.fak")
#load_pha("PCA/src_01_refake_1e4_a=2_NH=10.fak")
#load_pha("PCA/src_01b_refake_1e4_a=1_NH=01.fak")
#load_pha("PCA/refake_0934_2_23_1e5.fak")
#load_pha("PCA/refake_0934_1_21_1e5.fak")
#load_pha("PCA/refake_0934_2_23_1e4.fak")
#load_pha("PCA/refake_0934_1_21_1e4.fak")

arf1=unpack_arf("PCA/quiet.arf")
#arf1=unpack_arf("PCA/quiet_0934.arf")
rmf1=unpack_rmf("PCA/ccdid7_default.rmf")

#set_model("xsphabs.abs1*powlaw1d.p1")
#set_model("xsphabs.abs1*xspowerlaw.p1")
set_model("xswabs.abs1*xspowerlaw.p1")

### text for background model (old)
#bkg_model = "xsconstant.constant*(polynom1d.bkg_mdl_p1+gauss1d.bkg_mdl_g1+gauss1d.bkg_mdl_g2+gauss1d.bkg_mdl_g3+gauss1d.bkg_mdl_g4+gauss1d.bkg_mdl_g5+gauss1d.bkg_mdl_g6)"
#set_bkg_model(bkg_model)

### this file sets the parameters for the background model, which is assumed fixed except for xsconstant.constant
#execfile("acis-s-bkg.py")

set_arf(arf1)
set_rmf(rmf1)
notice(0.3,7.)

set_stat("cash")
set_method("simplex")
fit()

### finding the mle for the parameters
fit(1)
fit(1)
covariance(1)

### storing the fit results in fr
fr = get_fit_results()

### storing the covariance results in cr
cr = get_covar_results()

### combining the pertinent pieces from fr and cr into a dictionary
p = {"parnames" : np.copy(fr.parnames), "covar" : cr.extra_output, "parvals": fr.parvals, "ids": fr.datasets }

#print "calculating covariance"
covariance()
p=get_parameter_info()
parnames = p['parnames']
ids = np.array(get_fit_results().datasets)

### setting the inputs for the mht function
mu = np.copy(p["parvals"])
covar = np.copy(p["covar"])
num_draws = 10000
num_subdraws = 1
log = np.array([False,False,False])
inv = np.array([False,False,False])

### defaultprior = True is highly recommended
defaultprior = True

### function applied to the parameter
priorshape = np.array([False,False,False]) ### this input is only used if default prior is false

### is the prior defined as a function on the real scale?
originalscale = np.array([True,True,True])  ### this input is only used if default prior is false

### proportion of draws that should be metropolis rather than independence chain metropolis-hastings
p_M=.5

d = mht(p["parnames"], mu, covar, num_draws, 4, ids, fixARF=False, fixARFname='PCA/quiet.arf', pragBayes=True, num_subiter=10, usePCAforARF=True, useMNforARF=False, MH=False, multmodes=False, log=log, inv=inv, defaultprior=defaultprior,
    priorshape=priorshape, originalscale=originalscale, verbose=False, scale=1, sigma_m=False, p_M=p_M)

    
### the first column of d is the log posterior
### the first row of d corresponds to the mode
### the draws are returned on the original scale, even if they were transformed for the purposes of the proposal distribution

write_draws(d, "mh_draws.out")

mhint68 = analyze_draws(d, parnames, 1,False,True,False)
mh_int95 = analyze_draws(d, parnames, 2,False,True,False)
covarint68 = covariance_ci(p['covar'],p['parvals'],parnames, 1, True)
covarint95 = covariance_ci(p['covar'],p['parvals'],parnames, 2, True)
