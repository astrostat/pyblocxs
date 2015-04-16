# metropolis multivariate normal fit
# cash statistic is -2*loglikelihood
# fitted value for mean and covariances for multivariate normal from sherpa

from ui import *
import numpy as np
from sherpa.astro.ui import *
import sherpa.astro.ui as ui
#import scipy.special as sp
from funcs import *
#from sherpa.models.parameter import ParameterError 
from sherpa.utils.err import ParameterErr
from pycrates import *

class PCA(object): # Read pca definition file
    def __init__(self, filename):
        self.filename = filename
        pc = read_file(filename)
        self.emethod = get_keyval(pc,"EMETHOD")
        self.cmethod = get_keyval(pc,"CMETHOD")
        self.errext = get_keyval(pc,"ERREXT")
        self.bias = copy_colvals(pc,'BIAS')
        ext2 = read_file(filename + '[3]')   # could also "%s[3]" % filename
        self.colnames = get_col_names(ext2)
        self.component = copy_colvals(ext2,'COMPONENT')
        self.fvariance = copy_colvals(ext2,'FVARIANCE')
        self.eigenval = copy_colvals(ext2,'EIGENVAL')
        self.eigenvec = copy_colvals(ext2,'EIGENVEC')  # ND array

class rmfPCA: # Read rmfpca definition file
    def __init__(self):
        self.default = np.loadtxt('rmfPCA/default.txt')
        self.zeroidx_1, self.zeroidx_2 = np.loadtxt('rmfPCA/zeroidx.txt',dtype=int,unpack=True)-1
        self.meanPara = np.loadtxt('rmfPCA/meanPara.txt')
        self.mainPerc = np.loadtxt('rmfPCA/mainPerc.txt')
        self.meanrmf = np.loadtxt('rmfPCA/meanrmf.txt')
        self.sysErr = np.loadtxt('rmfPCA/sysErr.txt')
        self.eigenVal = np.loadtxt('rmfPCA/eigenVal.txt')
        self.eigenVec = np.loadtxt('rmfPCA/eigenVec.txt')

def sim_rmf(rmfpca,defrmf):
    newrmf= np.zeros((1078,1024))
    ncomp = len(rmfpca.eigenVal)
    bias  = defrmf - rmfpca.default
    e     = np.random.standard_normal(ncomp)
    para  = rmfpca.meanPara + np.dot((e*np.sqrt(rmfpca.eigenVal)),rmfpca.eigenVec)
    for i in range(1078):
        var  = min(49,max(4,para[1078+i]))
        temp = normpdf(np.arange(1,1025,1), mean=para[i], var=var)
        temp[np.where(temp<1e-6)]=0
        temp=temp/sum(temp)*rmfpca.mainPerc[i]
        newrmf[i]=temp
    newrmf[rmfpca.zeroidx_1,rmfpca.zeroidx_2]=rmfpca.meanrmf[rmfpca.zeroidx_1,rmfpca.zeroidx_2]
    newrmf=newrmf+bias+rmfpca.sysErr
    newrmf[np.where(newrmf<1e-6)]=0
    return newrmf,e

def sim_rmf_alt(rmfpca,defrmf,ein,esig):
    newrmf= np.zeros((1078,1024))
    ncomp = len(rmfpca.eigenVal)
    bias  = defrmf - rmfpca.default
    e     = ein+esig*np.random.standard_normal(ncomp)
    para  = rmfpca.meanPara + np.dot((e*np.sqrt(rmfpca.eigenVal)),rmfpca.eigenVec)
    for i in range(1078):
        var  = min(49,max(4,para[1078+i]))
        temp = normpdf(np.arange(1,1025,1), mean=para[i], var=var)
        temp[np.where(temp<1e-6)]=0
        temp=temp/sum(temp)*rmfpca.mainPerc[i]
        newrmf[i]=temp
    newrmf[rmfpca.zeroidx_1,rmfpca.zeroidx_2]=rmfpca.meanrmf[rmfpca.zeroidx_1,rmfpca.zeroidx_2]
    newrmf=newrmf+bias+rmfpca.sysErr
    newrmf[np.where(newrmf<1e-6)]=0
    return newrmf,e

def change_rmf(rmf_matrix):
    n_grp=np.zeros(1078,dtype=np.uint64)
    f_chan=np.zeros(0,dtype=np.uint32)
    n_chan=np.zeros(0,dtype=np.uint32)
    matrix=np.zeros(0,dtype=np.float64)
    for i in range(1078):
        flag=0
        startidx=0
        for j in np.arange(1024,dtype=np.uint32):
            if flag==0 and rmf_matrix[i,j]!=0:
                n_grp[i]=n_grp[i]+1
                f_chan=np.append(f_chan,j)
                startidx=j
                flag=1
            if flag==1 and rmf_matrix[i,j]==0:
                n_chan=np.append(n_chan,j-startidx)
                matrix=np.append(matrix,rmf_matrix[i,startidx:j])
                flag=0
    return n_grp, f_chan, n_chan, matrix
                
            
        

def sim_arf_alt(pca, defspecresp, rrin, rrsig, n=100): # Simulate an arf
    ncomp = len(pca.component)
    ncomp = min(n,ncomp)
    newarf = pca.bias + defspecresp
    rrout = rrin + rrsig * np.random.standard_normal(ncomp)
    #print rrout
    for i in xrange(ncomp):
        tmp = rrout[i] * pca.eigenval[i] * pca.eigenvec[i,]
        newarf = newarf + tmp
    return newarf,rrout

def sim_arf(pca, defspecresp,n=100): # Simulate an arf
    ncomp = len(pca.component)
    ncomp = min(n,ncomp)
    newarf = pca.bias + defspecresp
    rrout = np.random.standard_normal(ncomp)
    for i in xrange(ncomp):
        tmp = rrout[i] *pca.eigenval[i] * pca.eigenvec[i,]
        newarf = newarf +  tmp
    return newarf,rrout

def normpdf(x, mean, var):
    var = float(var)
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = np.exp(-(x-float(mean))**2/(2*var))
    return num/denom

def dmvnorm(x, mu, sigma, log=True):
    if np.min( np.linalg.eigvalsh(sigma))<=0 :
        raise RuntimeError, "Error: sigma is not positive definite"
    if np.max( np.abs(sigma-sigma.T))>=1e-9 :
        raise RuntimeError, "Error: sigma is not symmetric"
        
    logdens = -mu.size/2.0*np.log(2*np.pi)- 1/2.0*np.log( np.linalg.det(sigma) )-1/2.0 * np.dot( x-mu, np.dot(np.linalg.inv(sigma), x-mu ) )
    if log:
        return logdens
    else:
        dens = exp( logdens )
        return dens

def factorial( x, log=True):
    if type(x) != int:
        raise RuntimeError, "Must be an integer"
    if x==0 or x==1:
        if log:
            return 0
        else:
            return 1
            
    if x > 1:
        y = range(1,x+1)
        if log:
            result = np.sum( np.log(y))
            return result
        else:
            result = np.prod( y )
            return result

gammln_cof = np.array([76.18009173, -86.50532033, 24.01409822,
    -1.231739516e0, 0.120858003e-2, -0.536382e-5])
gammln_stp = 2.50662827465


#============= Gamma, Incomplete Gamma ===========

def gammln(xx):
    """Logarithm of the gamma function."""
    if xx==1 or xx==2:
        return 0
    else:
        global gammln_cof, gammln_stp
        x = xx - 1.
        tmp = x + 5.5
        tmp = (x + 0.5)*np.log(tmp) - tmp
        ser = 1.
        for j in range(6):
            x = x + 1.
            ser = ser + gammln_cof[j]/x
        return tmp + np.log(gammln_stp*ser)


def dmvt( x, mu, sigma, df, log=True, norm=False):
    
    if np.min( np.linalg.eigvalsh(sigma))<=0 :
        raise RuntimeError, "Error: sigma is not positive definite"
    if np.max( np.abs(sigma-sigma.T))>=1e-9 :
        raise RuntimeError, "Error: sigma is not symmetric"
        
    p = mu.size
    logdens_unnorm = -.5*np.log( np.linalg.det(sigma) ) - (df+p)/2.0*np.log( df + np.dot( x-mu, np.dot(np.linalg.inv(sigma), x-mu ) ) )
    if log :
        if norm:
            logdens = logdens_unnorm + gammln( (df+p)/2. )-gammln(df/2.)-(p/2.)*np.log(np.pi)+(df/2.)*np.log(df)
            return logdens
        else:
            return logdens_unnorm
        
    else:
        if norm:
            logdens = logdens_unnorm + gammln( (df+p)/2. )-gammln(df/2.)-(p/2.)*np.log(np.pi)+(df/2.)*np.log(df)
            dens = np.exp( logdens )
            return dens
        else:
            dens_unnorm = np.exp( logdens_unnorm )
            return dens_unnorm

def _set_par_vals(parnames, parvals):
    "Sets the paramaters to the given values"

    for (parname,parval) in zip(parnames,parvals):
        ui.set_par(parname, parval)


def mht(parnames, mu, sigma, num_iter, df, ids, fixARF=False, fixARFname='PCA/quiet.arf', pragBayes=True, num_subiter=1, usePCAforARF=True, useMNforARF=False, MH=True, multmodes=False, log=False, inv=False, defaultprior=True, priorshape=False, originalscale=True, verbose=False, scale=1, sigma_m=False, p_M=.5, maxconsrej=100, savedraws=True, thin=1):
    

    """
    p_M is mixing proportion of Metropolis draws in the mixture of MH and Metropolis draws
    """
    if ui.get_stat_name() != "cash":
        raise RuntimeError, "Statistic must be cash, not %s" % ui.get_stat_name()

    fr = ui.get_fit_results()
    if ids == None:
        ids = fr.datasets
    
    prior=np.repeat( 1.0, parnames.size)
    priorshape = np.array(priorshape)
    originalscale = np.array(originalscale)
    # if not default prior, prior calculated at each iteration
    if defaultprior!=True:
        if priorshape.size!=parnames.size:
            raise RuntimeError, "If not using default prior, must specify a function for the prior on each parameter"
        if originalscale.size!=parnames.size:
            raise RuntimeError, "If not using default prior, must specify the scale on which the prior is defined for each parameter"
    
    jacobian = np.repeat( False, parnames.size)
    ### jacobian needed if transforming parameter but prior for parameter on original scale
    if defaultprior!=True:
        ### if log transformed but prior on original scale, jacobian for those parameters is needed
        if np.sum( log*originalscale ) > 0:
            jacobian[ log*originalscale ] = True
        if np.sum( inv*originalscale ) > 0:
            jacobian[ inv*originalscale ] = True
    
    log = np.array(log)
    if log.size==1:
        log = np.tile( log, parnames.size)
        
    inv = np.array(inv)
    if inv.size==1:
        inv = np.tile( inv, parnames.size)
        
    if np.sum( log*inv)>0:
        raise RuntimeError, "Cannot specify both log and inv transformation for the same parameter"
    if MH:
        print "Running Metropolis-Hastings"
    else:
        print "Running Metropolis and Metropolis-Hastings"
    if multmodes:
        print "Will add second mode if gets stuck"  
    
    logpost    = np.zeros( (1000+1,1) )
    post       = np.zeros( (1000+1,1) )
    idxes      = np.zeros( (num_iter+1,1) )
    acceptARF  = np.zeros( (num_iter+1,1) )
    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    
    idxes[0]  = 1
    acceptARF[0] = 0
    arf1=unpack_arf(fixARFname)
    set_arf(arf1)    
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)
    
    statistics[0] = -.5*calc_stat(*ids)
    
    # Setting for PCA
    if usePCAforARF==True:
        defarf=get_arf()
        defresp=get_arf().specresp
        mypca=PCA('PCA/aref_Cedge.fits')
        ncomp=len(mypca.component)
        
    if defaultprior!=True:
        x=np.copy(current)
        if np.sum(originalscale) < parnames.size:
            for j in range(parnames.size):
                if log[j]*(1-originalscale[j])>0:
                    x[j]=np.log( x[j])
                if inv[j]*(1-originalscale[j])>0:
                    x[j]=1.0/x[j]
        for par in range(0, parnames.size):
            prior[par] = eval_prior( x[par], priorshape[par])

    statistics[0] += np.sum( np.log( prior) )
    if np.sum(log*jacobian)>0:
        statistics[0] += np.sum( np.log( current[log*jacobian] ) )
    if np.sum(inv*jacobian)>0:
        statistics[0] += np.sum( 2.0*np.log( np.abs(current[inv*jacobian]) ) )

    # using delta method to create proposal distribution on log scale for selected parameters
    if np.sum(log)>0:
        logcovar = np.copy(sigma)
        logcovar[:,log]= logcovar[:,log]/mu[log]
        logcovar[log]= (logcovar[log].T/mu[log]).T
        sigma = np.copy(logcovar)
        mu[log]=np.log(mu[log])
        current[log]=np.log( current[log])
    
    # using delta method to create proposal distribution on inverse scale for selected parameters
    if np.sum(inv)>0:
        invcovar = np.copy(sigma)
        invcovar[:,inv]= invcovar[:,inv]/(-1.0*np.power(mu[inv],2))
        invcovar[inv]= (invcovar[inv].T/(-1.0*np.power(mu[inv],2))).T
        sigma = np.copy(invcovar)
        mu[inv]=1.0/(mu[inv])
        current[inv]=1.0/( current[inv])
        
    iterations[0] = np.copy(mu) 
    zero_vec = np.zeros(mu.size)
    rejections=0
    modes = 1
    if np.mean(sigma_m) == False:
        sigma_m=np.copy(sigma)
        
    if useMNforARF==True and usePCAforARF==True:
        useMNforARF=False
        print "*** WARNING: PCA is used to sample ARF! ***"
               
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        ##########################################
        ### DRAW arf GIVEN parameters and data ###
        ##########################################

        if fixARF==False:
            if usePCAforARF==True:
                if i==1:
                    old_arf = defarf.specresp
                    old_rr  = np.random.standard_normal(ncomp)

                reset(get_source())
                
                u = np.random.uniform(0,1,1)
                if pragBayes==False and u > 0.5:
                   new_arf,new_rr = sim_arf_alt(mypca,defresp,old_rr,0.1)
                   print("m for arf") 
                else:
                   if pragBayes==False:
                      new_arf,new_rr = sim_arf(mypca,defresp, old_rr, 0.1)
                      print("accep for arf")
                   else:
                      new_arf,new_rr = sim_arf(mypca,defresp, old_rr, 1)

                defarf.specresp=new_arf
                stat_temp = -.5*calc_stat(*ids)
                
                accept_pr = 0
                if pragBayes==False:
                    mu0  = np.repeat(0,ncomp)
                    sig0 = np.diag(np.repeat(1,ncomp))
                    accept_pr += dmvnorm(new_rr,mu0,sig0)-dmvnorm(old_rr,mu0,sig0) 
                    accept_pr += stat_temp - statistics[i-1]           
                accept_pr = np.exp( accept_pr )
        
                u = np.random.uniform(0,1,1)
                if accept_pr > u or pragBayes==True or i==1:
                    old_arf = new_arf
                    old_rr  = new_rr
                    statistics[i-1] = np.copy(stat_temp)
                    acceptARF[i] = acceptARF[i-1] + 1
                    print("new arf")
                else:
                    print("old arf")
                    reset(get_source())
                    defarf.specresp=old_arf
                    acceptARF[i] = np.copy(acceptARF[i-1])
                        
            else:
                if pragBayes==False and useMNforARF==True:
                    maxidx = 1
                    for idx in range(1,1001,1):
                        if idx < 10:
                            arfname = 'ARFs/quiet_000%g' % idx + '.arf'
                        elif idx < 100:
                            arfname = 'ARFs/quiet_00%g' % idx + '.arf'
                        elif idx < 1000:
                            arfname = 'ARFs/quiet_0%g' % idx + '.arf'
                        else:
                            arfname = 'ARFs/quiet_%g' % idx + '.arf'
            
                        currentARF = unpack_arf(arfname)
                        set_arf(currentARF)
                        logpost[idx] = -.5*calc_stat(*ids)
                        if logpost[idx] > logpost[maxidx]: 
                            maxidx = idx 

                    for idx in range(1,1001,1):
                        post[idx] = np.exp(logpost[idx]-logpost[maxidx])
                    prob_sum = 0
                    for idx in range(1,1001,1):
                        prob_sum += post[idx]
                    for idx in range(1,1001,1):
                        post[idx] = post[idx]/prob_sum
                    prob_sum = 0    
                    u = np.random.uniform(0,1,1)
                    for idx in range(1,1001,1):
                        prob_sum += post[idx]
                        if prob_sum > u:
                            break

                    idxes[i] = idx        
                    if idx < 10:
                        arfname = 'ARFs/quiet_000%g' % idx + '.arf'
                    elif idx < 100:
                        arfname = 'ARFs/quiet_00%g' % idx + '.arf'
                    elif idx < 1000:
                        arfname = 'ARFs/quiet_0%g' % idx + '.arf'
                    else:
                        arfname = 'ARFs/quiet_%g' % idx + '.arf'

                    arf1 = unpack_arf(arfname)
                    set_arf(arf1)
                    stat_temp = -.5*calc_stat(*ids)
                
                    statistics[i-1] = np.copy(stat_temp)
                    acceptARF[i] = acceptARF[i-1] + 1
                
                else:
                    currentARF = arf1
        
                    idxes[i] = np.ceil(np.random.uniform(0,1,1)*1000)
        
                    if idxes[i] < 10:
                        arfname = 'ARFs/quiet_000%g' % idxes[i] + '.arf'
                    elif idxes[i] < 100:
                        arfname = 'ARFs/quiet_00%g' % idxes[i] + '.arf'
                    elif idxes[i] < 1000:
                        arfname = 'ARFs/quiet_0%g' % idxes[i] + '.arf'
                    else:
                        arfname = 'ARFs/quiet_%g' % idxes[i] + '.arf'
            
                    proposalARF = unpack_arf(arfname)
                    set_arf(proposalARF)
                    stat_temp = -.5*calc_stat(*ids)

                    accept_pr = np.exp( stat_temp - statistics[i-1] )
        
                    u = np.random.uniform(0,1,1)
                    if accept_pr > u or pragBayes==True:
                        arf1 = proposalARF
                        statistics[i-1] = np.copy(stat_temp)
                        acceptARF[i] = acceptARF[i-1] + 1
                        prev_arfname = arfname
                    else:
                        idxes[i] = np.copy(idxes[i-1])
                        acceptARF[i] = np.copy(acceptARF[i-1])
                        arfname = prev_arfname
                
                    set_arf(arf1)
            
            fit()
            covariance()
            tmp = get_parameter_info()
            mu = np.copy(tmp["parvals"])
            covar = np.copy(tmp["covar"])

            if usePCAforARF==False:
                print "Name of ARF used here = "+arfname

	else:
		print "Name of ARF used here = "+fixARFname
        
	##########################################
        ### DRAW parameters GIVEN arf and data ###
        ##########################################

        if fixARF==True:
            num_subiterations = 1
        else:
            num_subiterations = num_subiter
            
        for ii in range(num_subiterations):
            current = iterations[i-1]
            if verbose:
                if np.mod(i,1000)==0:
                    print "draw "+str(i)
            q = np.random.chisquare(df, 1)[0]
            if MH:
                while True:
                    try:
                        if modes==1 :
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
                        else:
                            u = np.random.uniform(0,1,1)
                            if u <= p :
                                proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
                            else:
                                proposal = mu2 + np.random.multivariate_normal(zero_vec, sigma2)/ np.sqrt(q/df)
                        if np.sum(log)>0:
                            proposal[log]=np.exp(proposal[log])
                        if np.sum(inv)>0:
                            proposal[inv]=1.0/proposal[inv]
                        
                        _set_par_vals(parnames, proposal)
                        break
                    except ParameterErr:
                        pass

                if defaultprior!=True:
                    x=np.copy(proposal)
                    ### is prior for all parameters evaluated on the original scale?
                    if np.sum(originalscale) < parnames.size:
                        for j in range(parnames.size):
                            if log[j]*(1-originalscale[j])>0:
                                x[j]=np.log( x[j])
                            if inv[j]*(1-originalscale[j])>0:
                                x[j]=1.0/x[j]
                    for par in range(0, parnames.size):
                        prior[par] = eval_prior( x[par], priorshape[par])
                
                #putting parameters back on log scale
                if np.sum(log)>0:
                    proposal[log] = np.log( proposal[log] )
                #putting parameters back on inverse scale
                if np.sum(inv)>0:
                    proposal[inv] = 1.0/proposal[inv]
                
                stat_temp = -.5*calc_stat(*ids)

                stat_temp += np.sum( np.log( prior))
                # adding jacobian (if necessary) with parameters on the log scale sum( log(theta)), but everything stored on log scale
                if np.sum(log*jacobian)>0:                  
                    stat_temp += np.sum( proposal[log*jacobian] )
                # adding jacobian (if necessary) with parameters on the inverse scale, sum(2*log(theta))=-sum(2*log(phi)), 
                if np.sum(inv*jacobian)>0:
                    stat_temp -= np.sum( 2.0*np.log( np.abs(proposal[inv*jacobian]) ) )
        
                if modes==1 :
                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
                else:
                    alpha = np.exp( stat_temp + evalmixture(current, mu, mu2, sigma, sigma2, df,p) - statistics[i-1] - evalmixture(proposal, mu, mu2, sigma, sigma2, df, p) )
            

            else:
                u = np.random.uniform(0,1,1)
                #if np.mod(i,2)==1:
                if u <= p_M:
                    print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            #if ((sigma==sigma_m).all()):
                            proposal = iterations[i-1] + np.random.multivariate_normal(zero_vec, sigma_m*scale)/ np.sqrt(q/df)
                            if np.sum(log)>0:
                                proposal[log]=np.exp(proposal[log])
                            if np.sum(inv)>0:
                                proposal[inv]=1.0/proposal[inv]
    
                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass
                    if defaultprior!=True:
                        x=np.copy(proposal)
                        ### is prior for all parameters evaluated on original scale?
                        if np.sum(originalscale) < parnames.size:
                            for j in range(parnames.size):
                                if log[j]*(1-originalscale[j])>0:
                                    x[j]=np.log( x[j])
                                if inv[i]*(1-originalscale[j])>0:
                                    x[j]=1.0/x[j]
                        for par in range(0, parnames.size):
                            prior[par] = eval_prior( x[par], priorshape[par])
                    if np.sum(log)>0:
                        proposal[log] = np.log( proposal[log] )
                    if np.sum(inv)>0:
                        proposal[inv]=1.0/proposal[inv]
                
                    stat_temp = -.5*calc_stat(*ids)
                    stat_temp += np.sum( np.log(prior))
                
                    # adding jacobian (if necessary) with parameters on the log scale sum( log(theta)), but everything stored on log scale
                    if np.sum(log*jacobian)>0:                  
                        stat_temp += np.sum( proposal[log*jacobian] )
                    # adding jacobian (if necessary) with parameters on the inverse scale, sum(2*log(theta))=-sum(2*log(phi)), 
                    if np.sum(inv*jacobian)>0:
                        stat_temp -= np.sum( 2.0*np.log( np.abs(proposal[inv*jacobian]) ) )

                    alpha = np.exp( stat_temp - statistics[i-1])

                else:
                    #MH jumping rule
                    print("mh for para")
                    while True:
                        try:
                            if modes==1 :
                                proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
                            else:
                                u = np.random.uniform(0,1,1)
                                if u <= p :
                                    proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
                                else:
                                    proposal = mu2 + np.random.multivariate_normal(zero_vec, sigma2)/ np.sqrt(q/df)
                            # back transform proposal for evaluation on original scale
                            if np.sum(log)>0:
                                proposal[log]=np.exp(proposal[log])
                            if np.sum(inv)>0:
                                proposal[inv]=1.0/proposal[inv]
                            
                            _set_par_vals(parnames, proposal)
                        
                            break
                        except ParameterErr:
                            pass
                    if defaultprior!=True:
                        x=np.copy(proposal)
                        ### is prior for all parameters evaluated on original scale
                        if np.sum(originalscale) < parnames.size:
                            for j in range(parnames.size):
                                if log[j]*(1-originalscale[j])>0:
                                    x[j]=np.log( x[j])
                                if inv[j]*(1-originalscale[j])>0:
                                    x[j]=1.0/x[j]
                        for par in range(0, parnames.size):
                            prior[par] = eval_prior( x[par], priorshape[par])
                    # transform parameter
                    if np.sum(log)>0:
                        proposal[log] = np.log( proposal[log] )
                    if np.sum(inv)>0:
                        proposal[inv] = 1.0/proposal[inv]
                    
                    stat_temp = -.5*calc_stat(*ids)
                    stat_temp += np.sum( np.log( prior))
                    # adding jacobian (if necessary) with parameters on the log scale sum( log(theta)), but everything stored on log scale
                    if np.sum(log*jacobian)>0:                  
                        stat_temp += np.sum( proposal[log*jacobian] )
                    # adding jacobian (if necessary) with parameters on the inverse scale, sum(2*log(theta))=-sum(2*log(phi)), 
                    if np.sum(inv*jacobian)>0:
                        stat_temp -= np.sum( 2.0*np.log( np.abs(proposal[inv*jacobian]) ) )
                
                    if modes==1 :
                        alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )

                    else:
                        alpha = np.exp( stat_temp + evalmixture(current, mu, mu2, sigma, sigma2, df,p) - statistics[i-1] - evalmixture(proposal, mu, mu2, sigma, sigma2, df, p) )
                                
            u = np.random.uniform(0,1,1)
            if u <= alpha:
                print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)
                rejections=0
            else:
                print("reject new para")
                iterations[i]=np.copy(iterations[i-1])
                statistics[i]=np.copy(statistics[i-1])
            
                ### added for test
                rejections += 1
                if ( rejections > maxconsrej and modes==1 and multmodes ):
                    print "need a second mode"
                    modes=2
                    mu2 = iterations[i]
                    print mu2
                    if sigma.size > 1:
                        eigenvals = np.linalg.eigvalsh(sigma)
                    else:
                        eigenvals = sigma
                    sigma2 = np.diag( np.repeat(max(eigenvals),mu.size) )
                    #sigma2 = sigma
                    d1 = np.exp(-.5*statistics[0])
                    d2 = np.exp(-.5*statistics[i])
                
                    ### returning normalized densities (not log)
                    t11 = dmvt( mu, mu, sigma, df, False, True)
                    t21 = dmvt( mu, mu2, sigma2, df, False, True)
                    t12 = dmvt( mu2, mu, sigma, df, False, True)
                    t22 = dmvt( mu2, mu2, sigma2, df, False, True)
                
                
                    ### p calculated with sigma2
                    #a = np.array([[d1[0],t21-t11],[d2[0],t22-t12]])
                    #b = np.array([t21,t22])
                    #x = np.linalg.solve( a, b)
                    #p = x[1]
                
                    p = d1/(d1+d2)
                    papprox = 1/ ( np.exp(np.log(d2)-np.log(d1)+np.log(t11)-np.log(t22)) +1)
                
                    print "p is" 
                    print p
                    print "p approx is"
                    print papprox
    
    if np.sum(log)>0:
        iterations[:,log] = np.exp( iterations[:,log])
    if np.sum(inv)>0:
        iterations[:,inv] = 1.0/( iterations[:,inv])
            
    result = np.hstack( (statistics,iterations,idxes,acceptARF) )
    return result





def mhttest(mu,sigma,num_iter, df, ptruth, dist, multmodes=True):
    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    iterations[0] = mu
    current=mu
    
    ### added for test
    mu2truth = mu + np.dot( np.repeat(dist,mu.size) , sigma )
    if sigma.size > 1:
        eigenvals = np.linalg.eigvalsh(sigma)
    else:
        eigenvals = sigma
    sigma2truth = np.diag( np.repeat(max(eigenvals),mu.size) )
    
    ###set_par("abs1.nh",current[0])
    ###set_par("p1.gamma",current[1])
    ###set_par("p1.ampl",current[2])
    
    #statistics[0] = calc_stat()
    
    ### added for test
    statistics[0] = evalmixture(current, mu, mu2truth, sigma, sigma2truth, df, ptruth)

    zero_vec = np.zeros(mu.size)
    
    ### added for test
    rejections=0
    modes = 1
    
    for i in range(1,num_iter+1,1):
        current = iterations[i-1]
        q = np.random.chisquare(df, 1)[0]
        if modes==1 :
            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
        ### added for test
        else:
            u = np.random.uniform(0,1,1)
            if u <= p :
                proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)
            else:
                proposal = mu2 + np.random.multivariate_normal(zero_vec, sigma2)/ np.sqrt(q/df)
                
        ###set_par("abs1.nh",proposal[0])
        ###set_par("p1.gamma",proposal[1])
        ###set_par("p1.ampl",proposal[2])
        ###stat_temp = calc_stat()
        
        ### added for test
        stat_temp = evalmixture( proposal, mu, mu2truth, sigma, sigma2truth, df, ptruth)
        
        if modes==1 :
            #alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
            alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )

        else:
            #alpha = np.exp( stat_temp + p*dmvt(current, mu, sigma, df, True, True)+(1-p)*dmvt(current, mu2, sigma2, df, True, True) - statistics[i-1] - p*dmvt(proposal, mu, sigma, df, True, True)-(1-p)*dmvt(proposal, mu2, sigma2, df, True, True) )
            alpha = np.exp( stat_temp + evalmixture(current, mu, mu2, sigma, sigma2, df,p) - statistics[i-1] - evalmixture(proposal, mu, mu2, sigma, sigma2, df, p) )
            
        u = np.random.uniform(0,1,1)
        if u <= alpha:
            iterations[i]=proposal
            statistics[i]=stat_temp
            rejections=0
        else:
            
            iterations[i]=iterations[i-1]
            statistics[i]=statistics[i-1]
            
            ### added for test
            rejections += 1
            if ( rejections >100 and modes==1 and multmodes ):
                print "need a second mode"
                modes=2
                mu2 = iterations[i]
                print mu2
                if sigma.size > 1:
                    eigenvals = np.linalg.eigvalsh(sigma)
                else:
                    eigenvals = sigma
                sigma2 = np.diag( np.repeat(max(eigenvals),mu.size) )
                d1 = np.exp(statistics[0])
                d2 = np.exp(statistics[i])
                t11 = dmvt( mu, mu, sigma, df, False, True)
                t21 = dmvt( mu, mu2, sigma2, df, False, True)
                t12 = dmvt( mu2, mu, sigma, df, False, True)
                t22 = dmvt( mu2, mu2, sigma2, df, False, True)
                a = np.array( [[t11,t21],[t12,t22]] )
                b = np.array( [d1,d2] )
                #assumed posterior not normalized, thus solving for pvec/k
                x = np.linalg.solve( a, b)
                # normalize to find p
                p = x[0]/np.sum(x)
                
                ### alternative calculation
                
                a = np.array([[d1[0],t21-t11],[d2[0],t22-t12]])
                b = np.array([t21,t22])
                x = np.linalg.solve( a, b)
                p2 = x[1]
                
                papprox = 1/ ( np.exp(np.log(d2)-np.log(d1)+np.log(t11)-np.log(t22)) +1)
                print "p is" 
                print p
                print "p constrained between 0 and 1 is "
                print p2
                print "p approx is"
                print papprox
                
                
    result = np.hstack( (statistics,iterations) )
    return result

def comp_prop(statistics, max_stat, cutoffs):
    ncutoffs = len(cutoffs)
    prop = np.zeros(ncutoffs)
    for i in range(ncutoffs):
        prop[i] = mean( statistics < max_stat+cutoffs[i])
    return prop
        
    
def evalmixture( current, mu1, mu2, sigma1, sigma2, df, p, log=True ):
    post = p*dmvt( current, mu1, sigma1, df, False, True ) + (1-p)*dmvt( current, mu2, sigma2, df, False, True)
    if log:
        logpost = np.log( post )
        return logpost
    return post


### test 1
### suppose the posterior distribution were exactly a mixture of two multivariate t's
### further, suppose that we know the mean of the major mode is mu and sigma 
def write_draws(draws, outfile):
    "Writes the draws to the file called outfile"
    fout = open(outfile, "w")
    for line in draws:
        for element in line:
            fout.write("%s " % element)
        fout.write("\n")
    fout.close()
    
#mu = np.array([0,0])
#sigma = np.array([[1,0],[0,2]])
#result = mhttest( mu, sigma, 10000, 4 , .9, 8,True)
#write_draws(result, "C:\Users\Jason\outfile")

######################
######################
######################


def mh_sampling(parnames, mu, sigma, num_iter, df, ids, fixARFname='PCA/quiet.arf', improvedBayes= False, num_within=10, num_subiter=10, p_M=.5, comp=8, p_M_arf=.5, sd_arf=.1, thin=1, scale=1):
    
    """
    ##p_M is mixing proportion of MH draws in the mixture of MH and Metropolis parameter draws
    ##p_M=0, all m draws; p_M=1, all mh draws
    """
    
    """
    ##p_M_arf is mixing proportion of MH draws in the mixture of MH and Metropolis arf draws
    ##p_M_arf=0, all m draws; p_M_arf=1, all mh draws
    """
    if ui.get_stat_name() != "cash":
        raise RuntimeError, "Statistic must be cash, not %s" % ui.get_stat_name()

    fr = ui.get_fit_results()
    if ids == None:
        ids = fr.datasets

    #Set up ARF
    arf1=unpack_arf(fixARFname)
    set_arf(arf1)
    
########################
### Fixed ARF Method ###
########################
       
    print "Fixed ARF Method"
    
    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )   
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)
    
    statistics[0] = -.5*calc_stat(*ids)
    iterations[0] = np.copy(mu)
    
    zero_vec = np.zeros(mu.size)
 
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

       	print "Name of ARF used here = "+fixARFname
       	
	##########################################
        ### DRAW parameters GIVEN arf and data ###
        ##########################################  

        current = iterations[i-1]
        q = np.random.chisquare(df, 1)[0]
        u = np.random.uniform(0,1,1)
        if u > p_M:
                    ##print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            proposal = iterations[i-1] + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass

                    stat_temp = -.5*calc_stat(*ids)
                           
                    alpha = np.exp( stat_temp - statistics[i-1])
             
        else:

                    #MH jumping rule
                    ##print("mh for para")
                    while True: 
                       try:
                            
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                        
                            break
                       except ParameterErr:
                            pass

                    
                    stat_temp = -.5*calc_stat(*ids)

                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
                                
        u = np.random.uniform(0,1,1)
        
        if u <= alpha:
                ###print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)

        else:
                ###print("reject new para")
                iterations[i]=np.copy(iterations[i-1])
                statistics[i]=np.copy(statistics[i-1])

    result1 = np.hstack( (statistics,iterations) )

########################
### Pragmatic Method ###
########################

    print "Pragmatic Method"

    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)

    stat_temp = -.5*calc_stat(*ids)
    statistics[0] = stat_temp
    
    # Setting for PCA
    defarf=get_arf()
    defresp=get_arf().specresp
    mypca=PCA('PCA/aref_Cedge.fits')
    #mypca=PCA('RXJ1856/aref_hrcsletg_8.fits')
    ncomp=len(mypca.component)
    ncomp=min(ncomp,comp)
    
    arfcomp = np.zeros( (num_iter+1,ncomp) )
        
    iterations[0] = np.copy(mu) 
    zero_vec = np.zeros(mu.size)

    new_rr=np.zeros(ncomp)
    new_arf=defresp
        
    num_subiterations = num_subiter

    if improvedBayes==False:
        num_within=1
 
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        ##########################################
        ### DRAW arf GIVEN parameters and data ###
        ##########################################

        reset(get_source())
        # reset(get_model())
        defarf.specresp=new_arf
        
        if i%num_within==0:

            new_arf,new_rr = sim_arf(mypca,defresp,n=ncomp)
            defarf=get_arf()
            defarf.specresp=new_arf
            # arf=get_arf()
            stat_temp = -.5*calc_stat(*ids)

            fit()
            covariance()
            tmp = get_parameter_info()
            mu = np.copy(tmp["parvals"])
            covar = np.copy(tmp["covar"])

        arfcomp[i] = np.copy(new_rr)

	print "First component of ARF used here = "+ str(new_rr[0])

        
	##########################################
        ### DRAW parameters GIVEN arf and data ###
        ##########################################  


        current = iterations[i-1]
        _set_par_vals(parnames, current)
        stat_prev = -.5*calc_stat(*ids)
        
        for ii in range(num_subiterations):

            q = np.random.chisquare(df, 1)[0]
            u = np.random.uniform(0,1,1)
            if u > p_M:
                    ##print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            proposal = current + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass

                    stat_temp = -.5*calc_stat(*ids)
                           
                    alpha = np.exp( stat_temp - stat_prev)
             
            else:


                    #MH jumping rule
                    ##print("mh for para")
                    while True: 
                       try:
                            
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                        
                            break
                       except ParameterErr:
                            pass

                    
                    stat_temp = -.5*calc_stat(*ids)

                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - stat_prev - dmvt(proposal, mu, sigma, df) )
                                
            u = np.random.uniform(0,1,1)
            if u <= alpha:
                ###print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)
                current=proposal
                stat_prev=stat_temp
            else:
                ###print("reject new para")
                iterations[i]=np.copy(current)
                statistics[i]=np.copy(stat_prev)

    result2 = np.hstack( (statistics,iterations,arfcomp) )

########################
### FullyBays Method ###
########################
    print "FullyBays Method"

    target=np.hstack((iterations,arfcomp))
    target=target.transpose()
    bigsigma=np.cov(target)
    bigmean=target.mean(1)
    
    mu1=bigmean[:mu.size]
    sig11=bigsigma[:mu.size,:mu.size]
    sig12=bigsigma[:mu.size,mu.size:]
    sig22_inv=np.linalg.inv(bigsigma[mu.size:,mu.size:])
    con_var=scale*(sig11-np.dot(np.dot(sig12,sig22_inv),sig12.transpose()))
    
    startindex=1
    
    stat_prev=statistics[startindex]
    current=iterations[startindex]
    arfcomp_current=arfcomp[startindex]

    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    arfcomp = np.zeros( (num_iter+1,ncomp) )
    
    iterations[0]=current
    statistics[0]=stat_prev
    arfcomp[0]=arfcomp_current

    #con_prob_prev=dmvnorm(current,mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_current.transpose()),con_var)

    con_prob_prev=100

    mu0  = np.repeat(0,ncomp)
    sig0 = np.diag(np.repeat(1,ncomp))
    prior_prev=dmvnorm(arfcomp_current,mu0,sig0)
    # oldarf=arf

    # index=0
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        print "First component of ARF used here = "+ str(arfcomp_current[0])

        for j in range(0,thin,1):
            
            u = np.random.uniform(0,1,1)
            if u > p_M_arf:
                #index=0
                #print("mh for para")
                ##Mh jumping rule

                reset(get_source())
                #reset(get_model())

                new_arf, arfcomp_proposal= sim_arf(mypca,defresp,n=ncomp)
                defarf=get_arf()
                defarf.specresp=new_arf
                # oldarf=arf
                # arf=get_arf()
                
                while True:
                      try:
                            proposal=np.random.multivariate_normal(mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)

                            _set_par_vals(parnames, proposal)
                                
                            break
                      except ParameterErr:
                            pass
                            
                con_prob_proposal=dmvnorm(proposal,mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)
                prior_proposal=dmvnorm(arfcomp_proposal,mu0,sig0)

                stat_proposal = -.5*calc_stat(*ids)
                               
                alpha = np.exp( stat_proposal - stat_prev + con_prob_prev - con_prob_proposal)

            else:
                #index=1
                #M jumping rule
                #print("m for para")
                reset(get_source())
                #reset(get_model())

                new_arf, arfcomp_proposal= sim_arf_alt(mypca,defresp,arfcomp_current,sd_arf,n=ncomp)
                defarf=get_arf()
                defarf.specresp=new_arf
                # oldarf=arf
                # arf=get_arf()
                
                while True:
                      try:
                            proposal=np.random.multivariate_normal(mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)

                            _set_par_vals(parnames, proposal)
                                
                            break
                      except ParameterErr:
                            pass
                            
                con_prob_proposal=dmvnorm(proposal,mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)
                prior_proposal=dmvnorm(arfcomp_proposal,mu0,sig0)

                stat_proposal = -.5*calc_stat(*ids)
                               
                alpha = np.exp( stat_proposal - stat_prev + con_prob_prev - con_prob_proposal + prior_proposal - prior_prev)

            u = np.random.uniform(0,1,1)
            
            if u <= alpha:
                #print("accept new para")
                current=proposal
                arfcomp_current=arfcomp_proposal
                stat_prev=stat_proposal
                con_prob_prev=con_prob_proposal
                prior_prev=prior_proposal
                #print(index)
                #oldarf=arf
        #arf=oldarf
        iterations[i]=np.copy(current)
        statistics[i]=np.copy(stat_prev)
        arfcomp[i]=np.copy(arfcomp_current)

    result3= np.hstack( (statistics,iterations,arfcomp) )

    result=np.hstack( (result1,result2,result3) )

    return result


######################
######################
######################


def mh_sampling_newdata(parnames, mu, sigma, num_iter, df, ids, fixARFname='PCA/quiet.arf', improvedBayes= False, num_within=10, num_subiter=10, p_M=.5, comp=8, p_M_arf=.5, sd_arf=.1, thin=1, scale=1):
    
    """
    ##p_M is mixing proportion of MH draws in the mixture of MH and Metropolis parameter draws
    ##p_M=0, all m draws; p_M=1, all mh draws
    """
    
    """
    ##p_M_arf is mixing proportion of MH draws in the mixture of MH and Metropolis arf draws
    ##p_M_arf=0, all m draws; p_M_arf=1, all mh draws
    """
    if ui.get_stat_name() != "cash":
        raise RuntimeError, "Statistic must be cash, not %s" % ui.get_stat_name()

    fr = ui.get_fit_results()
    if ids == None:
        ids = fr.datasets

    ##Set up ARF
    #arf1=unpack_arf(fixARFname)
    #set_arf(arf1)
    
########################
### Fixed ARF Method ###
########################
       
    print "Fixed ARF Method"
    
    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )   
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)
    
    statistics[0] = -.5*calc_stat(*ids)
    iterations[0] = np.copy(mu)
    
    zero_vec = np.zeros(mu.size)
 
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

       	print "Name of ARF used here = "+fixARFname
       	
	##########################################
        ### DRAW parameters GIVEN arf and data ###
        ##########################################  

        current = iterations[i-1]
        q = np.random.chisquare(df, 1)[0]
        u = np.random.uniform(0,1,1)
        if u > p_M:
                    ##print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            proposal = iterations[i-1] + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass

                    stat_temp = -.5*calc_stat(*ids)
                           
                    alpha = np.exp( stat_temp - statistics[i-1])
             
        else:

                    #MH jumping rule
                    ##print("mh for para")
                    while True: 
                       try:
                            
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                        
                            break
                       except ParameterErr:
                            pass

                    
                    stat_temp = -.5*calc_stat(*ids)

                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
                                
        u = np.random.uniform(0,1,1)
        
        if u <= alpha:
                ###print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)

        else:
                ###print("reject new para")
                iterations[i]=np.copy(iterations[i-1])
                statistics[i]=np.copy(statistics[i-1])

    result1 = np.hstack( (statistics,iterations) )

########################
### Pragmatic Method ###
########################

    print "Pragmatic Method"

    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)

    stat_temp = -.5*calc_stat(*ids)
    statistics[0] = stat_temp
    
    # Setting for PCA
    defarf=get_arf()
    defresp=get_arf().specresp
    #mypca=PCA('PCA/aref_Cedge.fits')
    mypca=PCA('RXJ1856/aref_hrcsletg_8.fits')
    ncomp=len(mypca.component)
    ncomp=min(ncomp,comp)
    
    arfcomp = np.zeros( (num_iter+1,ncomp) )
        
    iterations[0] = np.copy(mu) 
    zero_vec = np.zeros(mu.size)

    new_rr=np.zeros(ncomp)
    new_arf=defresp
        
    num_subiterations = num_subiter

    if improvedBayes==False:
        num_within=1
 
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        ##########################################
        ### DRAW arf GIVEN parameters and data ###
        ##########################################
        # reset(get_source())
        # reset(get_model())
        defarf.specresp=new_arf
        
        if i%num_within==0:

            new_arf,new_rr = sim_arf(mypca,defresp,n=ncomp)
            defarf=get_arf()
            defarf.specresp=new_arf
            # arf=get_arf()
            stat_temp = -.5*calc_stat(*ids)

            fit()
            covariance()
            tmp = get_parameter_info()
            mu = np.copy(tmp["parvals"])
            covar = np.copy(tmp["covar"])

        arfcomp[i] = np.copy(new_rr)

	print "First component of ARF used here = "+ str(new_rr[0])

        
	##########################################
        ### DRAW parameters GIVEN arf and data ###
        ##########################################  


        current = iterations[i-1]
        _set_par_vals(parnames, current)
        stat_prev = -.5*calc_stat(*ids)
        
        for ii in range(num_subiterations):

            q = np.random.chisquare(df, 1)[0]
            u = np.random.uniform(0,1,1)
            if u > p_M:
                    ##print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            proposal = current + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass

                    stat_temp = -.5*calc_stat(*ids)
                           
                    alpha = np.exp( stat_temp - stat_prev)
             
            else:


                    #MH jumping rule
                    ##print("mh for para")
                    while True: 
                       try:
                            
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                        
                            break
                       except ParameterErr:
                            pass

                    
                    stat_temp = -.5*calc_stat(*ids)

                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - stat_prev - dmvt(proposal, mu, sigma, df) )
                                
            u = np.random.uniform(0,1,1)
            if u <= alpha:
                ###print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)
                current=proposal
                stat_prev=stat_temp
            else:
                ###print("reject new para")
                iterations[i]=np.copy(current)
                statistics[i]=np.copy(stat_prev)

    result2 = np.hstack( (statistics,iterations,arfcomp) )

########################
### FullyBays Method ###
########################
    print "FullyBays Method"

    target=np.hstack((iterations,arfcomp))
    target=target.transpose()
    bigsigma=np.cov(target)
    bigmean=target.mean(1)
    
    mu1=bigmean[:mu.size]
    sig11=bigsigma[:mu.size,:mu.size]
    sig12=bigsigma[:mu.size,mu.size:]
    sig22_inv=np.linalg.inv(bigsigma[mu.size:,mu.size:])
    con_var=scale*(sig11-np.dot(np.dot(sig12,sig22_inv),sig12.transpose()))
    
    startindex=1
    
    stat_prev=statistics[startindex]
    current=iterations[startindex]
    arfcomp_current=arfcomp[startindex]

    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    arfcomp = np.zeros( (num_iter+1,ncomp) )
    
    iterations[0]=current
    statistics[0]=stat_prev
    arfcomp[0]=arfcomp_current

    #con_prob_prev=dmvnorm(current,mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_current.transpose()),con_var)

    con_prob_prev=100

    mu0  = np.repeat(0,ncomp)
    sig0 = np.diag(np.repeat(1,ncomp))
    prior_prev=dmvnorm(arfcomp_current,mu0,sig0)
    # oldarf=arf

    # index=0
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        print "First component of ARF used here = "+ str(arfcomp_current[0])

        for j in range(0,thin,1):
            
            u = np.random.uniform(0,1,1)
            if u > p_M_arf:
                #index=0
                #print("mh for para")
                ##Mh jumping rule

                #reset(get_source())
                #reset(get_model())

                new_arf, arfcomp_proposal= sim_arf(mypca,defresp,n=ncomp)
                defarf=get_arf()
                defarf.specresp=new_arf
                # oldarf=arf
                # arf=get_arf()
                
                while True:
                      try:
                            proposal=np.random.multivariate_normal(mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)

                            _set_par_vals(parnames, proposal)
                                
                            break
                      except ParameterErr:
                            pass
                            
                con_prob_proposal=dmvnorm(proposal,mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)
                prior_proposal=dmvnorm(arfcomp_proposal,mu0,sig0)

                stat_proposal = -.5*calc_stat(*ids)
                               
                alpha = np.exp( stat_proposal - stat_prev + con_prob_prev - con_prob_proposal)

            else:
                #index=1
                #M jumping rule
                #print("m for para")
                #reset(get_source())
                #reset(get_model())

                new_arf, arfcomp_proposal= sim_arf_alt(mypca,defresp,arfcomp_current,sd_arf,n=ncomp)
                defarf=get_arf()
                defarf.specresp=new_arf
                # oldarf=arf
                # arf=get_arf()
                
                while True:
                      try:
                            proposal=np.random.multivariate_normal(mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)

                            _set_par_vals(parnames, proposal)
                                
                            break
                      except ParameterErr:
                            pass
                            
                con_prob_proposal=dmvnorm(proposal,mu1+np.dot(np.dot(sig12,sig22_inv),arfcomp_proposal.transpose()),con_var)
                prior_proposal=dmvnorm(arfcomp_proposal,mu0,sig0)

                stat_proposal = -.5*calc_stat(*ids)
                               
                alpha = np.exp( stat_proposal - stat_prev + con_prob_prev - con_prob_proposal + prior_proposal - prior_prev)

            u = np.random.uniform(0,1,1)
            
            if u <= alpha:
                #print("accept new para")
                current=proposal
                arfcomp_current=arfcomp_proposal
                stat_prev=stat_proposal
                con_prob_prev=con_prob_proposal
                prior_prev=prior_proposal
                #print(index)
                #oldarf=arf
        #arf=oldarf
        iterations[i]=np.copy(current)
        statistics[i]=np.copy(stat_prev)
        arfcomp[i]=np.copy(arfcomp_current)

    result3= np.hstack( (statistics,iterations,arfcomp) )

    result=np.hstack( (result1,result2,result3) )

    return result






########rmf########

def mh_sampling_rmf(parnames, mu, sigma, num_iter, df, ids, fixRMFname='rmfPCA/default_ccdid7.rmf', improvedBayes= False, num_within=10, num_subiter=10, p_M=.5, comp=4, p_M_rmf=.8, sd_rmf=.1, thin=10, scale=0.1):
    
    """
    ##p_M is mixing proportion of MH draws in the mixture of MH and Metropolis parameter draws
    ##p_M=0, all m draws; p_M=1, all mh draws
    """
    
    """
    ##p_M_rmf is mixing proportion of MH draws in the mixture of MH and Metropolis arf draws
    ##p_M_rmf=0, all m draws; p_M_rmf=1, all mh draws
    """
    if ui.get_stat_name() != "cash":
        raise RuntimeError, "Statistic must be cash, not %s" % ui.get_stat_name()

    fr = ui.get_fit_results()
    if ids == None:
        ids = fr.datasets

    ##Set up RMF
    rmf1=unpack_rmf(fixRMFname)
    set_rmf(rmf1)
    
########################
### Fixed RMF Method ###
########################
       
    print "Fixed RMF Method"
    
    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )   
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)
    
    statistics[0] = -.5*calc_stat(*ids)
    iterations[0] = np.copy(mu)
    
    zero_vec = np.zeros(mu.size)
 
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

       	print "Name of RMF used here = "+fixRMFname
       	
	##########################################
        ### DRAW parameters GIVEN RMF and data ###
        ##########################################  

        current = iterations[i-1]
        q = np.random.chisquare(df, 1)[0]
        u = np.random.uniform(0,1,1)
        if u > p_M:
                    ##print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            proposal = iterations[i-1] + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass

                    stat_temp = -.5*calc_stat(*ids)
                           
                    alpha = np.exp( stat_temp - statistics[i-1])
             
        else:

                    #MH jumping rule
                    ##print("mh for para")
                    while True: 
                       try:
                            
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                        
                            break
                       except ParameterErr:
                            pass

                    
                    stat_temp = -.5*calc_stat(*ids)

                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - statistics[i-1] - dmvt(proposal, mu, sigma, df) )
                                
        u = np.random.uniform(0,1,1)
        
        if u <= alpha:
                ###print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)

        else:
                ###print("reject new para")
                iterations[i]=np.copy(iterations[i-1])
                statistics[i]=np.copy(statistics[i-1])

    result1 = np.hstack( (statistics,iterations) )

########################
### Pragmatic Method ###
########################

    print "Pragmatic Method"

    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    
    current=np.copy(mu)
    _set_par_vals(parnames, current)

    stat_temp = -.5*calc_stat(*ids)
    statistics[0] = stat_temp
    
    # change fixed RMF to matrix format
    defrmf_matrix = np.zeros( (1078, 1024) )
    defrmf=get_rmf()
    def_n_grp=defrmf.n_grp
    def_f_chan=defrmf.f_chan
    def_n_chan=defrmf.n_chan
    def_matrix=defrmf.matrix
    point=0
    point_grp=0
    for row in range(0,1078,1):
          for grp in range(1, int(def_n_grp[row]+1),1):
              defrmf_matrix[row, int(def_f_chan[point_grp]):int((def_f_chan[point_grp]+def_n_chan[point_grp]))] = def_matrix[point:int((point+def_n_chan[point_grp]))]
              point = point + def_n_chan[point_grp]
              point_grp = point_grp + 1

    myrmfpca=rmfPCA()
    ncomp=len(myrmfpca.eigenVal)
    ncomp=min(ncomp,comp)
    myrmfpca.eigenVec=myrmfpca.eigenVec[:ncomp,:]
    myrmfpca.eigenVal=myrmfpca.eigenVal[:ncomp]

    rmfcomp = np.zeros( (num_iter+1,ncomp) )
        
    iterations[0] = np.copy(mu) 
    zero_vec = np.zeros(mu.size)

    new_rr=np.zeros(ncomp)
    new_n_grp=def_n_grp
    new_f_chan=def_f_chan
    new_n_chan=def_n_chan
    new_matrix=def_matrix
        
    num_subiterations = num_subiter

    if improvedBayes==False:
        num_within=1
 
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        ##########################################
        ### DRAW rmf GIVEN parameters and data ###
        ##########################################
        reset(get_source())
        defrmf.n_grp=new_n_grp
        defrmf.f_chan=new_f_chan
        defrmf.n_chan=new_n_chan
        defrmf.matrix=new_matrix
        
        if i%num_within==0:

            new_rmf_matrix,new_rr = sim_rmf(myrmfpca,defrmf_matrix)
            new_n_grp,new_f_chan,new_n_chan,new_matrix=change_rmf(new_rmf_matrix)
            defrmf.n_grp=new_n_grp
            defrmf.f_chan=new_f_chan
            defrmf.n_chan=new_n_chan
            defrmf.matrix=new_matrix
            stat_temp = -.5*calc_stat(*ids)

            fit()
            covariance()
            tmp = get_parameter_info()
            mu = np.copy(tmp["parvals"])
            covar = np.copy(tmp["covar"])

        rmfcomp[i] = np.copy(new_rr)

	print "First component of RMF used here = "+ str(new_rr[0])

        
	##########################################
        ### DRAW parameters GIVEN rmf and data ###
        ##########################################  


        current = iterations[i-1]
        _set_par_vals(parnames, current)
        stat_prev = -.5*calc_stat(*ids)
        
        for ii in range(num_subiterations):

            q = np.random.chisquare(df, 1)[0]
            u = np.random.uniform(0,1,1)
            if u > p_M:
                    ##print("m for para")
                    #Metropolis jumping rule
                    while True:
                        try:
                            proposal = current + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                            
                            break
                        except ParameterErr:
                            pass

                    stat_temp = -.5*calc_stat(*ids)
                           
                    alpha = np.exp( stat_temp - stat_prev)
             
            else:


                    #MH jumping rule
                    ##print("mh for para")
                    while True: 
                       try:
                            
                            proposal = mu + np.random.multivariate_normal(zero_vec, sigma)/ np.sqrt(q/df)

                            _set_par_vals(parnames, proposal)
                        
                            break
                       except ParameterErr:
                            pass

                    
                    stat_temp = -.5*calc_stat(*ids)

                    alpha = np.exp( stat_temp + dmvt(current, mu, sigma, df) - stat_prev - dmvt(proposal, mu, sigma, df) )
                                
            u = np.random.uniform(0,1,1)
            if u <= alpha:
                ###print("accept new para")
                iterations[i]=np.copy(proposal)
                statistics[i]=np.copy(stat_temp)
                current=proposal
                stat_prev=stat_temp
            else:
                ###print("reject new para")
                iterations[i]=np.copy(current)
                statistics[i]=np.copy(stat_prev)

    result2 = np.hstack( (statistics,iterations,rmfcomp) )

########################
### FullyBays Method ###
########################
    print "FullyBays Method"

    target=np.hstack((iterations,rmfcomp))
    target=target.transpose()
    bigsigma=np.cov(target)
    bigmean=target.mean(1)
    
    mu1=bigmean[:mu.size]
    sig11=bigsigma[:mu.size,:mu.size]
    sig12=bigsigma[:mu.size,mu.size:]
    sig22_inv=np.linalg.inv(bigsigma[mu.size:,mu.size:])
    con_var=scale*(sig11-np.dot(np.dot(sig12,sig22_inv),sig12.transpose()))
    
    startindex=1
    
    stat_prev=statistics[startindex]
    current=iterations[startindex]
    rmfcomp_current=rmfcomp[startindex]

    iterations = np.zeros( (num_iter+1, mu.size) )
    statistics = np.zeros( (num_iter+1,1) )
    rmfcomp = np.zeros( (num_iter+1,ncomp) )
    
    iterations[0]=current
    statistics[0]=stat_prev
    rmfcomp[0]=rmfcomp_current

    #con_prob_prev=dmvnorm(current,mu1+np.dot(np.dot(sig12,sig22_inv),rmfcomp_current.transpose()),con_var)

    con_prob_prev=100

    mu0  = np.repeat(0,ncomp)
    sig0 = np.diag(np.repeat(1,ncomp))
    prior_prev=dmvnorm(rmfcomp_current,mu0,sig0)

    # index=0
    for i in range(1,num_iter+1,1):
        print "------------------------------------------------------------"
        print "Beginning iteration "+str(i)

        print "First component of RMF used here = "+ str(rmfcomp_current[0])

        for j in range(0,thin,1):
            
            u = np.random.uniform(0,1,1)
            if u > p_M_rmf:
                #index=0
                #print("mh for para")
                ##Mh jumping rule

                reset(get_source())

                #new_arf, arfcomp_proposal= sim_arf(mypca,defresp,n=ncomp)                
                #defarf.specresp=new_arf


                new_rmf_matrix,rmfcomp_proposal = sim_rmf(myrmfpca,defrmf_matrix)
                new_n_grp,new_f_chan,new_n_chan,new_matrix=change_rmf(new_rmf_matrix)
                defrmf.n_grp=new_n_grp
                defrmf.f_chan=new_f_chan
                defrmf.n_chan=new_n_chan
                defrmf.matrix=new_matrix
                
                while True:
                      try:
                            proposal=np.random.multivariate_normal(mu1+np.dot(np.dot(sig12,sig22_inv),rmfcomp_proposal.transpose()),con_var)

                            _set_par_vals(parnames, proposal)
                                
                            break
                      except ParameterErr:
                            pass
                            
                con_prob_proposal=dmvnorm(proposal,mu1+np.dot(np.dot(sig12,sig22_inv),rmfcomp_proposal.transpose()),con_var)
                prior_proposal=dmvnorm(rmfcomp_proposal,mu0,sig0)

                stat_proposal = -.5*calc_stat(*ids)
                               
                alpha = np.exp( stat_proposal - stat_prev + con_prob_prev - con_prob_proposal)

            else:
                #index=1
                #M jumping rule
                #print("m for para")
                reset(get_source())

                new_rmf_matrix, rmfcomp_proposal= sim_rmf_alt(myrmfpca,defrmf_matrix,rmfcomp_current,sd_rmf)                
                new_n_grp,new_f_chan,new_n_chan,new_matrix=change_rmf(new_rmf_matrix)
                defrmf.n_grp=new_n_grp
                defrmf.f_chan=new_f_chan
                defrmf.n_chan=new_n_chan
                defrmf.matrix=new_matrix
                
                while True:
                      try:
                            proposal=np.random.multivariate_normal(mu1+np.dot(np.dot(sig12,sig22_inv),rmfcomp_proposal.transpose()),con_var)

                            _set_par_vals(parnames, proposal)
                                
                            break
                      except ParameterErr:
                            pass
                            
                con_prob_proposal=dmvnorm(proposal,mu1+np.dot(np.dot(sig12,sig22_inv),rmfcomp_proposal.transpose()),con_var)
                prior_proposal=dmvnorm(rmfcomp_proposal,mu0,sig0)

                stat_proposal = -.5*calc_stat(*ids)
                               
                alpha = np.exp( stat_proposal - stat_prev + con_prob_prev - con_prob_proposal + prior_proposal - prior_prev)

            u = np.random.uniform(0,1,1)
            
            if u <= alpha:
                #print("accept new para")
                current=proposal
                rmfcomp_current=rmfcomp_proposal
                stat_prev=stat_proposal
                con_prob_prev=con_prob_proposal
                prior_prev=prior_proposal
                #print(index)

        iterations[i]=np.copy(current)
        statistics[i]=np.copy(stat_prev)
        rmfcomp[i]=np.copy(rmfcomp_current)

    result3= np.hstack( (statistics,iterations,rmfcomp) )

    result=np.hstack( (result1,result2,result3) )


    return result
