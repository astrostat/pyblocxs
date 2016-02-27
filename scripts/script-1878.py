from ui import *

import sherpa.astro.ui as ui

from mhtest import *

execfile("funcs.py")





load_pha("Calibration/1878new/spec1878.pi")

energy_range = {"lo":.5,"hi": 7}
notice_id(1,**energy_range)

set_stat("cash")
set_method("simplex")
set_model("xsphabs.abs1*(xsvapec.kT1+xsvapec.kT2)") 

thaw(kT1.O)
thaw(kT1.Ne)
thaw(kT1.Fe)

abs1.nH = 0.335
kT1.kT = 0.153 ; kT1.kT.min = 0.01 ; kT1.kT.max = 0.5
kT2.kT = 0.491 ; kT2.kT.min = 0.3 ; kT2.kT.max = 4.0
kT1.norm = 0.046
kT2.norm = 0.003

kT1.C = kT1.O
kT1.N = kT1.O
kT1.Ni = kT1.Fe
kT1.Mg = kT1.Fe
kT1.Si = kT1.Fe
kT1.Ca = kT1.Fe

kT2.O = kT1.O
kT2.Ne = kT1.Ne
kT2.Fe = kT1.Fe

kT2.C = kT2.O
kT2.N = kT2.O
kT2.Ni = kT2.Fe
kT2.Mg = kT2.Fe
kT2.Si = kT2.Fe
kT2.Ca = kT2.Fe

## text for background model, use the constant bkg arf

get_bkg_arf().specresp=get_bkg_arf().specresp * 0 + 100. 
set_bkg_model("const1d.bkg_c1*(powlaw1d.bkg_pow1+ powlaw1d.bkg_pow2+gauss1d.bkg_g1+gauss1d.bkg_g2+gauss1d.bkg_g3+gauss1d.bkg_g4+gauss1d.bkg_g5+gauss1d.bkg_g6)")

execfile("Calibration/1878new/bkg_model_1878.py")

fit()
fit()

covariance(1)

##proj(1)



### storing the fit results in fr

fr = get_fit_results()

### storing the covariance results in cr

cr = get_covar_results()

### combining the pertinent pieces from fr and cr into a dictionary

p = {"parnames" : np.copy(fr.parnames), "covar" : cr.extra_output, "parvals": fr.parvals, "ids": fr.datasets }



mu = np.copy(p['parvals'])

sigma = np.copy(p['covar'])

num_draws = 3000

ids = np.copy(p['ids'])


d = mh_sampling(p["parnames"], mu , sigma ,num_draws, 4, ids, fixARFname='Calibration/1878/spec1878.corr.arf', improvedBayes=True, p_M_arf=0.5, comp=8, sd_arf=.1, thin=10)


write_draws(d, "realdata/new1878.out")

