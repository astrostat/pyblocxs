"""
Metropolis multivariate normal fit.

Cash statistic is -2 * log(likelihood)

Based on Jason's original mh.py code.
"""

import numpy as np
import sherpa.astro.ui as ui
from sherpa.utils.err import ParameterErr

__all__ = ["mht", "dmvt", "dmvnorm", "draw_t", "accept_tcash"]

def dmvnorm(x, mu, sigma, log=True):
    logdens = -mu.size/2.0*log(2*pi)- 1/2.0*np.log( np.linalg.det(sigma) )-1/2.0 * np.dot( x-mu, np.dot(np.linalg.inv(sigma), x-mu ) )
    if log:
        return logdens
    else:
        return np.exp(logdens)

def dmvt(x, mu, sigma, df, log=True, norm=False):
    
    logdens_unnorm = -.5*np.log( np.linalg.det(sigma) ) - (df+mu.size)/2.0*np.log( df + np.dot( x-mu, np.dot(np.linalg.inv(sigma), x-mu ) ) )
    if log:
        if norm:
            raise RuntimeError, "Not yet implemented!"
        else:
            return logdens_unnorm
    else:
        return np.exp(logdens_unnorm)

def _set_par_vals(parnames, parvals):
    "Sets the paramaters to the given values"

    for (parname,parval) in zip(parnames,parvals):
        ui.set_par(parname, parval)

def draw_t(mu, current, sigma, df=4, **kwargs):
    """Create a new set of parameter values using the t distribution.

    Given the best-guess (mu) and current (current) set of
    parameters, along with the covariance matrix (sigma),
    return a new set of parameters.
    """

    zero_vec = np.zeros_like(mu)
    q = np.random.chisquare(df, 1)[0]
    proposal = mu + np.random.multivariate_normal(zero_vec, sigma) / np.sqrt(q/df)
    return proposal

def accept_tcash(current, current_stat, proposal, proposal_stat,
                mu, sigma, df=4, **kwargs):
    """Should the proposal be accepted (using the Cash statistic and the t distribution)?
    """

    alpha = np.exp( -0.5*proposal_stat + dmvt(current, mu, sigma, df) + 0.5*current_stat - dmvt(proposal, mu, sigma, df) )
    u = np.random.uniform(0,1,1)
    return u <= alpha

def mht(parnames, mu, sigma, niter=1000, id=None,
        file=None, verbose=True, normalize=True,
        draw=draw_t, accept=accept_tcash, **kwargs):
    """Metropolis-Hastings.

    The default distribution is the t distribution, and the statistic is
    assumed to be the Cash statistic.

    The kwargs are passed through to the draw and accept routines.

    The draw routine is used to create a new proposal.
    The accept routine is used to determine whether to accept the proposal.

    If verbose is True then the iteration results are printed to STDOUT
    after each iteration. If file is not None then the iteration results
    are printed to the given file wach iteration. If normalize is True
    then the displayed results (whether to STDOUT or file) are relative
    to the best-fit values rather than absolute ones (so the values for
    the xpos parameter are written out as xpos-xpos_0 where xpos_0 is
    the value from the input mu argument). This also holds for the
    statistic value (so the results are statistic-statistic_0). The
    reason for normalize is to try and avoid lose of information
    without having to display numbers to 15 decimal places.
    """

    # Should we just change to cash here instead of throwing an error?
    #
    if ui.get_stat_name() != "cash":
        raise RuntimeError, "Statistic must be cash, not %s" % ui.get_stat_name()

    if id == None:
        idval = ui.get_default_id()
    else:
        idval = id

    # Output storage
    #
    nelem = niter + 1
    npars = mu.size
    if npars != len(parnames):
        raise RuntimeError, "mu.size = %d  len(parnames) = %d!" % (npars, len(parnames))

    params = np.zeros((nelem,npars))
    stats  = np.zeros(nelem)
    alphas = np.zeros(nelem)

    # Using a bool is technically nicer, but stick with an int8 here for easier processing
    # of the output.
    #
    ##acceptflag = np.zeros(nelem, dtype=np.bool)
    acceptflag = np.zeros(nelem, dtype=np.int8)

    params[0] = mu.copy()
    current = mu.copy()
    alphas[0] = 0

    _set_par_vals(parnames, current)
    stats[0] = ui.calc_stat(id=idval)

    if normalize:
        outstr = "# iteration accept d_statistic %s" % " d_".join(parnames)
    else:
        outstr = "# iteration accept statistic %s" % " ".join(parnames)
    if verbose:
        print outstr
    if file != None:
	fout = open(file, "w")
    	fout.write(outstr)
        fout.write("\n")

    def draw_to_string(idx):
        "Return the given draw as a string for display/output"
        if normalize:
            outstr = "%-6d %1d %g %s" % (idx, acceptflag[idx], stats[idx]-stats[0], " ".join(["%g" % (v-v0) for (v,v0) in zip(params[idx],params[0])]))
        else:
            outstr = "%-6d %1d %g %s" % (idx, acceptflag[idx], stats[idx], alphas[idx], " ".join(["%g" % v for v in params[idx]]))
        return outstr

    # Iterations
    # - no burn in at present
    # - the 0th element of the params array is the input value
    # - we loop until all parameters are within the allowable
    #   range; should there be some check to ensure we are not
    #   rejecting a huge number of proposals, which would indicate
    #   that the limits need increasing or very low s/n data?
    #
    for i in range(1,nelem,1):

        current = params[i-1]

        # Create a proposal and set the parameter values. If any lie
        # outside the allowed range then catch this (ParameterError)
        # and create a new proposal.
        #
        while True:
            try:
                proposal = draw(mu, current, sigma, **kwargs)
                _set_par_vals(parnames, proposal)
                break
            except ParameterErr:
                pass

        # Do we accept this proposal?
        #
        stat_temp = ui.calc_stat(id=idval)
        alphas[i] = np.exp( -0.5*stat_temp + dmvt(current, mu, sigma, 4) + 0.5*stats[i-1] - dmvt(proposal, mu, sigma, 4) )

        if accept(current, stats[i-1], proposal, stat_temp,
                  mu, sigma, **kwargs):
            params[i] = proposal.copy()
            stats[i]  = stat_temp
            acceptflag[i] = 1
        else:
            params[i] = params[i-1]
            stats[i]  = stats[i-1]
            acceptflag[i] = 0

        outstr = draw_to_string(i)
    if verbose:
        print outstr
    if file != None:
            fout.write(outstr)
            fout.write("\n")

    if file != None:
	fout.close()
        print "Created: %s" % file

    # Return a dictionary containing the draws
    #
    out = { "parnames": parnames, "statistic": stats, "accept": acceptflag, "alphas": alphas, "iteration": np.arange(0,nelem,1) }
    for (idx,name) in zip(range(npars),parnames):
    	if out.has_key(name):
            raise RuntimeError, "Unexpected name clash: parameter '%s'" % name
        out[name] = params[:,idx]
    return out
