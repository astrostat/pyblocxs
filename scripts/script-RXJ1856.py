from ui import *

import sherpa.astro.ui as ui

from mhtest import *

execfile("funcs.py")


# load the pha, rmf, and background files
# need to exclude BACKGROUND_* columns to avoid clashing with background pha file

load_data("RXJ1856/p1_113+3380+3381+3382+3399+15293+14418_HRC-LEG_1_xspec.pha[cols CHANNEL,COUNTS,BIN_LO,BIN_HI]")
load_rmf("RXJ1856/hrcf00113_repro_leg_p1.rmf")
load_bkg("RXJ1856/p1_113+3380+3381+3382+3399+15293+14418_HRC-LEG_1_bkg.pha")

# carry out the analysis is wavelength (Angstrom) space
set_analysis("wave")

# define the wavelength range over which to carry out the analysis
# ignore the range (59.5,68) to exclude the part affected by the chip gap, which appears to be not well-modeled by the gARF
ignore()
notice(25.,59.5) ; notice(68.,80.)

# plot_data()

# set statistic and method

set_stat("cash")
set_method("simplex")

# set_model("xsphabs.abs1*xsbbody.bb1")

# get default ARF
arf = get_arf()

# set background ARF to be flat
get_bkg_arf().specresp = get_bkg_arf().specresp * 0 + 1.0
bkg_arf = get_bkg_arf()

# read background scale
bkg_scale = get_bkg_scale()

# set full source model, including background model
set_full_model(arf("xsphabs.abs1*xsbbody.bb1") + bkg_scale*bkg_arf("const1d.bkg_c1 * polynom1d.pp"))
bkg_c1.c0 = 78.195

abs1.nH = 0.00791702
set_par(bb1.kt, frozen=False, val=0.0620321, min=0.001, max=0.2)
set_par(bb1.norm, frozen=False, val=0.000285548)

# define background model parameters
set_bkg_full_model(bkg_arf("polynom1d.pp"))
freeze(pp)
pp.c0=-2.08129 ; pp.c1=0.349143 ; pp.c2=-0.0246803 ; pp.c3=0.000968759 ; pp.c4=-2.312e-5 ; pp.c5=3.4386e-7 ; pp.c6=-3.11659e-9 ; pp.c7=1.57691e-11 ; pp.c8=-3.41824e-14

fit()



##get_covar_results().extra_output

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



p_M=0.5

ids = np.copy(p['ids'])



d = mh_sampling_newdata(p["parnames"], mu , sigma ,num_draws, 4, ids, fixARFname='RXJ1856/p1_113+3380+3381+3382+3399+15293+14418_HRC-LEG_1.arf', improvedBayes=True, p_M_arf=0.5, comp=8, sd_arf=.1, scale=1)


write_draws(d, "RXJ1856.out")
