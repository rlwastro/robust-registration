"""Robust cross-matching with astrometric correction

The main function is robust_ring(), which cross-matches two catalogs using
the robust Bayesian algorithm and returns information on matched sources
plus the infinitesimal rotation vector omega that shifts and rotates the
astrometry.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# global variables
# convert units from arcseconds to radians
arc2rad = 3600 * 180 / np.pi

def getRC(cat, ref, pairs, mid=True):
    """Generate the paired catalogs from the input based on the pairing index. 
    
    :param cat:    Input catalog dataframe to be corrected, with (x,y,z) coordinates.
    :param ref:    Reference catalog dataframe with (x,y,z) coordinates. 
    :param pairs:  Index pairs.
    :param mid:    Boolean value, to indicate if the input catalog is to be corrected to the midpoints of
                   cat&ref. If False, input catalog is to be corrected to the referecne catalog.
    :type cat:     pandas.DataFrame
    :type ref:     pandas.DataFrame
    :type pairs:   numpy.ndarray
    :type mid:     bool
    :returns:      tuple of (r,c): where r is the coordinates of the catalog in the pair, and 
                   c is the coordinates of the reference in the pair.
    """
    if isinstance(cat, pd.DataFrame): 
        a = cat[list('xyz')].values
    else:
        a = cat
    
    if isinstance(ref, pd.DataFrame): 
        b = ref[list('xyz')].values
    else:
        b = ref
    
    r = a[pairs[:,0]]          # catalog coordinates
    
    if mid:
        c = 0.5*(r+b[pairs[:,1]])  # mid-point between the catalog and reference coordinates of the pair
    else:
        c = b[pairs[:,1]]
        
    return r, c


class SingularMatrixError(Exception):
    pass

def solveRC(r, c, sigma, w=None):
    """Solve for the 3D transformation vector omega based on the (r,c) catalog pairs. 
    The estimated omega is used to correct the source directions in the catalog r to 
    the refererence directions in the catalog c.
    
    :param r:      Input catalog coordinates in pairs to be corrected, (x,y,z) vectors.
    :param c:      Reference catalog coordinates in pairs, (x,y,z) vectors. 
    :param sigma:  Known uncertainty parameter of the catalog, to be used for estimation.
    :param w:      A vector of weights to be assigned to each of the paired sources in (r,c).
                   If w is given, the estimation method uses the robust-estimation algorithm.
                   If the w is None, the estimation method uses the Least-squares estimation algorithm.
    :type r:       numpy.ndarray
    :type c:       numpy.ndarray
    :type sigma:   float
    :type w:       numpy.array or None
    :returns:      (omega): estimated 3D transformation vector
    """
    sigmarad2 = np.square(sigma / arc2rad)
    
    if w is None:
        w = np.ones(r.shape[0]) # if no w is given, we set them to 1
    
    # Estimate omegas
    y = np.tensordot(w/sigmarad2,np.cross(r,c),axes=(0,0))
    M = -np.tensordot(w/sigmarad2,vdyadicp(r,r),axes=(0,0))
    np.fill_diagonal(M, M.diagonal()+(w/sigmarad2).sum())
    try:
        omega = np.linalg.solve(M,y)
    except np.linalg.LinAlgError as e:
        if 'singular' in str(e).lower():
            # convert to SingularMatrix error
            raise SingularMatrixError(str(e))
        raise

    return omega

def L2_est(cat, ref, pairs, sigma, mid=True):
    """
    Returns estimated omega by using the method of least-squares (not robust) based on Budavari and Lubow (2012).
    
    :param cat:    Input catalog dataframe with (x,y,z) coordinates. 
    :param ref:    Reference catalog dataframe with (x,y,z) coordinates. 
    :param pairs:  Index pairs. 
    :param sigma:  Astrometic uncertainty parameter of the catalog.
    :param mid:    Boolean value, to indicate if the input catalog is to be corrected to the midpoints of
                   cat&ref. If False, input catalog is to be corrected to the referecne catalog.
    :type cat:     pandas.DataFrame
    :type ref:     pandas.DataFrame
    :type pairs:   numpy.ndarray
    :type mid:     bool
    :returns:     (omega): 3D transformation vector of (x,y,z) based on the least-squares method
    """
    r,c = getRC(cat,ref,pairs,mid=True)
    return solveRC(r,c,sigma,w=None)

def getpairs(cat, ref, sep):
    """Match two lists of objects by position using 3-D Cartesian distances

    Matches directions in catalog with directions in reference, for distances within a circle of radius = sep.
    *All* matches within the circle are returned.  Input catalogs need not be sorted.
    
    This function:
    1) figures out if xyz coordinates have the largest range, and pick that for the most efficient pre-sort
    2) then match pairs with sorting
    
    :param cat:   Input catalog dataframe with (x,y,z) coordinates.
    :param ref:   Reference catalog dataframe with (x,y,z) coordinates. 
    :param sep:   Separation radius threshold. If the distance between a pair of sources 
                  less than sep, it is considered as a match.
    :type cat:    pandas.DataFrame or np.ndarray
    :type ref:    pandas.DataFrame or np.ndarray
    :type sep:    float
    :returns:     (p1, p2) index arrays into source lists with matching pairs
    
    Based on function boxMatch in starmatch_hist.py
    - 2019 February 2, Rick White
    """
    
    # check input data type
    if isinstance(cat, pd.DataFrame) and isinstance(ref, pd.DataFrame):
        xyz1 = cat[list('xyz')].values
        xyz2 = ref[list('xyz')].values
    else:
        xyz1 = cat
        xyz2 = ref
    
    if xyz1.ndim != 2 or xyz2.ndim != 2 or xyz1.shape[-1] != 3 or xyz2.shape[-1] != 3:
        raise ValueError("xyz1 and xyz2 parameters must be 2-D [*,3] arrays")
    n1 = xyz1.shape[0]
    n2 = xyz2.shape[0]
    if n1==0 or n2==0:
        # no sources in catalog, return empty pointer array
        return np.empty((0,2),dtype=int)

    # pick the coordinate with the largest range for initial sort
    range1 = xyz1.max(axis=0)-xyz1.min(axis=0)
    range2 = xyz2.max(axis=0)-xyz2.min(axis=0)
    
    zrange = range1.clip(min=range2) # call Rick if you need help with this XXXXXXXXXXXXXXXX
    isdim = zrange.argsort()
    saxis = isdim[-1] # sort along this axis
    r = xyz1[:,saxis] 
    c = xyz2[:,saxis]

    # Sort the arrays by increasing y-coordinate
    is1 = r.argsort()
    xyz1 = xyz1[is1]
    r = r[is1]
    is2 = c.argsort()
    xyz2 = xyz2[is2]
    c = c[is2]
    # find search limits in y2 for each object in y1
    # note this is designed to include the points that are exactly equal to maxdiff
    kvlo = c.searchsorted(r-sep,'left').clip(0, len(c))
    kvhi = c.searchsorted(r+sep,'right').clip(kvlo, len(c))

    # build a list of matching segments
    p1 = []
    p2 = []
    sepsq = sep**2
    for i in range(n1):
        klo = kvlo[i]
        khi = kvhi[i]
        dsepsq = np.square(xyz2[klo:khi,:]-xyz1[i]).sum(axis=-1)
        w = (dsepsq <= sepsq).nonzero()[0]
        if w.size > 0:
            p1.append(np.zeros(w.size,dtype=np.int)+i)
            p2.append(klo+w)
    if p1:
        return np.vstack([is1[np.hstack(p1)], is2[np.hstack(p2)]]).T
    else:
        return np.empty((0,2),dtype=int)

def computeWeights(r, c, omega, sigma, gamma, area, renormalize=True, tcut=30):
    """
    Compute weights in the robust estimation method. 
    The function returns a list of weights to be assigned to each of the paired 
    sources in catalogs r and c. The estimate is based on the inputs of
    omega, sigma, gamma and the footprint area of the image. 
    
    :param r:            Input catalog coordinates in pairs to be corrected, (x,y,z) vectors.
    :param c:            Reference catalog coordinates in pairs, (x,y,z) vectors.
    :param omega:        3D transformation omega vector.
    :param sigma:        Astrometric uncertainty parameter of the catalog, units in arcsec.
    :param gamma:        Fraction of good matches among all pairs.
    :param area:         Area of the catalog footprint in steradians. 
    :param renormalize:  Boolean value. If true, renormalize weights to avoid underflow.
    :param tcut:         Cut-off value for renormalization. Default is 30 (empirical value).
    :type r:             numpy.ndarray
    :type c:             numpy.ndarray
    :type omega:         numpy.ndarray
    :type sigma:         float
    :type gamma:         float
    :type area:          float
    :type renormalize:   bool
    :type tcut:          float
    :returns:            (w) weights: list of weights to be assigned to paired sources.
    """
    
    sigmarad2 = (sigma / arc2rad)**2
    alpha = area*gamma/sigmarad2/(2*np.pi)
    dvec = c - r - np.cross(omega,r)
    t2 = np.square(dvec).sum(axis=1)/sigmarad2
    
    if renormalize: #renormalize weights if all exponent values are large to avoid underflows
        tmin = t2.min()
        if tmin > tcut:
            t2 = t2 + (tcut-tmin)
    v = alpha * np.exp(-0.5 * t2)
    w = v / (v + 1-gamma)
    return w

def vdyadicp(a, b):
    """
    Returns dyadic product of the input arrays (a, b)
 
    :param a: [\*,3] input array.  May be multidimensional (e.g. 5,4,3) as long 
                                   as last dimension is 3.
    :param b: [\*,3] input array.  May be multidimensional (e.g. 5,4,3) as long 
                                   as last dimension is 3.
    :type a: numpy.ndarray
    :type b: numpy.ndarray
    :returns: [\*, 3, 3] array of outer products of a, b.
    """

    a = np.asarray(a)
    b = np.asarray(b)
    s = a.shape
    if s[-1] != 3:
        raise ValueError('a,b last dimension must be 3')
    if s != b.shape:
        raise ValueError('a,b must have same shape [...,3]')
    return np.einsum('...i,...j->...ij', a, b)

def rob_est(r, c, sigma, gamma, area, omega=None, w=None, sigma_init=None, niter=10, nextr=100, printerror=False, verbose=False):
    """
    Iterative estimation with intial sigma convergence of 'niter' iterations, followed by
    'nextr' steps of convergence with a fixed sigma value. The maximum number of iterations is niter+nextr.
    
    :param r:            Input catalog coordinates in pairs to be corrected, (x,y,z) vectors.
    :param c:            Reference catalog coordinates in pairs, (x,y,z) vectors. 
    :param sigma:        True value of sigma, the astrometric uncertainty of the catalog.
    :param gamma:        Fraction of good matches among all pairs.
    :param area:         Area of the footprint, units in steradians.
    :param omega:        An initial guess of the transformation vector. 
                         If not given, omega will be estimated using the least-squares algorithm.
    :param w:            A vector of weights to be assigned to each of the paired sources in (r,c).
                         If w is None, the intial step of the estimation takes w as all ones.
    :param sigma_init:   Assign a large initial value for sigma. If None, will use 25*sigma.
    :param niter:        Min number of iterations for the convergence.
    :param nextr:        Max number of additional iterations for the convergence.
    :param printerror:   Boolean value, if true then print error messages for ring failures.
    :param verbose:      Boolean value, if true prints info about progress
    :type r:             numpy.ndarray
    :type c:             numpy.ndarray
    :type sigma:         float
    :type gamma:         float
    :type area:          float
    :type sigma_init:    None or float
    :type omega:         float
    :type w:             float
    :type niter:         int 
    :type nextr:         int 
    :type printerror:    bool
    :type verbose:       bool
    :returns:            (omega, w) omega: 3D transformation vector estimated by robust algorithm
                         w: weights of the last iteration.
    """
    sigma_init = sigma_init or 25 * sigma
    
    # Specify an array of sigma values to use for
    # sigma convergence from initial to final values.
    sigma_array = sigma_init*(sigma/sigma_init)**(np.arange(niter, dtype=float)/(niter-1))
    if verbose:
        print(f"sigma init {sigma_init} final {sigma} gamma {gamma} niter {niter} nextr {nextr}")
    
    # use w=1 if too few pairs
    n = r.shape[0]
    if n < 3:
        if printerror:
            print(f"Skipping the weighting with {n} pairs")
        # estimate with weights of all ones
        omega = solveRC(r, c, sigma=sigma, w=None)
        w = np.ones(r.shape[0])
    
    else:
        # Nominal value for gamma to avoid problem
        if gamma > 0.9:
            gamma = 0.9
        
        # If not give an initial guess of omega, estimate assuming all pairs are "good"
        # This is the same as estimating omega using the least-squares method
        if omega is None:
            omega = solveRC(r, c, sigma=sigma, w=w)

        # Loop over a number of iterations
        for it in range(niter+nextr):

            i = min(it,sigma_array.size-1)

            # assign omega_t+1=omega_t
            omega_last = omega # ref

            # Estimate weights
            w = computeWeights(r=r, c=c, omega=omega, sigma=sigma_array[i], gamma=gamma, area=area)

            # Estimate omega_t using robust algorithm, with weights
            try:
                omega = solveRC(r, c, sigma=sigma_array[i], w=w)
            except SingularMatrixError as e:
                # singular matrix, just skip this ring
                # expand on error message
                raise SingularMatrixError("Singular matrix in the iteration {} wtsum {}".format(it+1,w.sum()))

            # Stopping criteria: |omega_{t+1} - omega_{t}| < epsilon
            diff = np.sqrt(((omega-omega_last)**2).sum())
            if verbose:
                print(f"{i} sigma {sigma_array[i]} diff {diff} wtsum {w.sum()}")
            if diff < 1.e-11 and it>niter:
                break
    return omega, w

def robust_ring(cat, ref, area, radius, sigma, minpairs=None, **kw):
    """
    Estimate omega with robust algorithm in rings. Obtain optimal estimate from best ring.
    This uses adaptive ring widths based on the number of pairs per ring.
    Additional keywords are passed to process_ring().
    
    :param cat:           Input catalog dataframe with (x,y,z) coordinates.
    :param ref:           Reference catalog dataframe with (x,y,z) coordinates.
    :param area:          Area of the footprint, units in steradians.
    :param radius:        Separation radius threshold.
    :param sigma:         True value of sigma, the astrometric uncertainty of the catalog.
    :param minpairs:      Number of pairs per ring (default is determined by getRing from cat sizes)
    :type cat:            pandas.DataFrame
    :type ref:            pandas.DataFrame
    :type area:           float
    :type radius:         float
    :type sigma:          float
    :type minpairs:       int
    :returns:             (bestomega, bestpairs, bestwt) omega: 3D transformation vector estimated in 
                          the optimal ring by robust algorithm,
                          bestpairs: pairs in the optimal ring,
                          bestwt: robust weights for bestpairs.
    """
    
    # get all pairs within search radius
    pairs = getpairs(cat, ref, radius/arc2rad)
    if pairs.shape[0] < 1:
        raise ValueError('robust_ring(): no pairs found')
    ringpairs = getRing(cat, ref, pairs)
    return process_ring(cat, ref, pairs, ringpairs, area, radius, sigma, **kw)

def getRing(cat, ref, pairs, minpairs=None):
    """
    Divide all pairs into rings with minpairs pairs and return a list of pairs in the rings. 
    Rings overlap by 1/2 minpairs, so they go from 0-minpairs, 
    0.5*minpairs-1.5*minpairs, minpairs-2*minpairs, etc.
    If minpairs is not specified, it is determined from the number of sources in 
    cat and ref.
    
    :param cat:       Input catalog dataframe with (x,y,z) coordinates.
    :param ref:       Reference catalog dataframe with (x,y,z) coordinates.
    :param pairs:     Match index of pairs within radius.
    :param minpairs:  Number of pairs per ring.  Default is 6*min(len(cat),len(ref)).
    :type cat:        pandas.DataFrame
    :type ref:        pandas.DataFrame
    :type pairs:      numpy.ndarray
    :type minpairs:   int
    :returns:         (rpairs): a list of arrays that are pair index within rings
    """
   
    # check input data type
    if isinstance(cat, pd.DataFrame) and isinstance(ref, pd.DataFrame):
        a = cat[list('xyz')].values
        b = ref[list('xyz')].values
    else:
        a = cat
        b = ref

    # fast partitioning of pairs
    if minpairs is None:
        minpairs = 6 * max(min(a.shape[0],b.shape[0]),5)
    sep = np.sqrt(((a[pairs[:,0]]-b[pairs[:,1]])**2).sum(axis=1)) *arc2rad #(180*3600/np.pi)
    ind = np.argsort(sep)
    sep = sep[ind]
    pairs = pairs[ind,:]
    
    # if there are fewer than minpairs pairs, there will be only a single ring
    # nrings = (pairs.shape[0]+minpairs//2-1) // (minpairs//2)
    nrings = (pairs.shape[0]-minpairs//4) // (minpairs//2)
    nrings = max(nrings,1)
    klo = np.arange(nrings,dtype=int)*(minpairs//2)
    khi = klo + (pairs.shape[0] - klo[-1])
    rpairs = [pairs[klo[iring]:khi[iring],:] for iring in range(nrings)]
    return rpairs

def process_ring(cat, ref, pairs, ringpairs, area, radius, sigma, sigma_init=None, gamma=None, niter=10, nextr=100, mid=True,
        printprogress=True, printerror=False):
    """
    Estimate omega with robust algorithm in rings. Obtain optimal estimate from best ring.
    Internal function to process pairs that are already split into rings.
    
    :param cat:           Input catalog dataframe with (x,y,z) coordinates.
    :param ref:           Reference catalog dataframe with (x,y,z) coordinates.
    :param area:          Area of the footprint, units in steradians.
    :param radius:        Separation radius threshold.
    :param sigma:         True value of sigma, the astrometric uncertainty of the catalog.
    :param sigma_init:    If not None, assign a large initial value for sigma.
    :param gamma:         Fraction of good matches among all pairs. If None, will be computed in estimation.
    :param niter:         Min number of iterations for the convergence.
    :param nextr:         Max number of additional iterations for the convergence.
    :param mid:           Boolean value, indicate if reference as midpoints of the two catalogs
    :param prinprogress:  Boolean value, if true shows progress bar.
    :param printerror:    Boolean value, indicate if track error.
    :type cat:            pandas.DataFrame
    :type ref:            pandas.DataFrame
    :type area:           float
    :type radius:         float
    :type sigma:          float
    :type sigma_init:     None or float
    :type gamma:          None or float
    :type niter:          int
    :type nextr:          int
    :type mid:            bool
    :type printprogress:  bool
    :type printerror:     bool
    :returns:             (bestomega, bestpairs, bestwt) omega: 3D transformation vector estimated in 
                          the optimal ring by robust algorithm,
                          bestpairs: pairs in the optimal ring,
                          bestwt: robust weights for bestpairs.
    """
    sigma_init = sigma_init or 25 * sigma # heuristic estimate for convergence parameter

    nrings = len(ringpairs)
    if printprogress:
        print(f"Split {pairs.shape[0]} pairs into {nrings} overlapping rings")
        print(f"process_ring: sigma {sigma} sigma_init {sigma_init}")

    # gamma = gamma or min(cat.shape[0],ref.shape[0]) / pairs.shape[0]
    if not gamma:
        # count just sources actually included in pairs
        # this makes a difference when search radius is small and many sources don't match
        n1 = (np.bincount(pairs[:,0])!=0).sum()
        n2 = (np.bincount(pairs[:,1])!=0).sum()
        gamma = min(n1,n2) / pairs.shape[0]

    # increase gamma because expected match is higher in the correct ring
    #gfac = pairs.shape[0] / np.mean([x.shape[0] for x in ringpairs])
    #gamma = gamma * gfac
    #if printprogress and gfac != 1:
    #    print(f"Increased gamma by factor {gfac:.2f} to {gamma}")
    
    # Initial best sum(weight)=0
    bestwtsum = 0.0
    bestomega = None
    bestring = nrings
    
    if printprogress:
        # print progress bar (but disable on non-TTY output)
        # disable = None
        # print progress bar
        disable = False
    else:
        # do not print progress bar
        disable = True
    sys.stdout.flush()
    loop = tqdm(total=nrings, position=0, leave=False, disable=disable)

    # loop over all pairs to find optimal omega estimate
    for iring in range(nrings):
        rpairs = ringpairs[iring]

        # paired catalog and reference in ring
        r,c = getRC(cat, ref, rpairs, mid)
    
        # estimate omega using robust algorithm
        try:
            omega, w = rob_est(r, c, sigma, gamma, area, sigma_init=sigma_init, niter=niter, nextr=nextr,
                    printerror=printerror, verbose=printprogress>1)
            
        except SingularMatrixError as e:
            if printerror:
                print(e)
                print('continuing to next ring')
            continue
            
        # Sum of weights is the number of good pairs
        wtsum = w.sum()
        if wtsum > bestwtsum:
            bestring = iring
            bestpairs = rpairs
            bestomega = omega
            bestwtsum = wtsum
            bestwt = w
        if not printerror:
            loop.set_description("Computing...".format(iring))
            loop.update(1)
    loop.close()
    if bestomega is None:
        if printerror:
            print("process_ring: no solution found")
        return np.zeros(3), np.zeros((0,2),dtype=int), np.zeros(0,dtype=float)
    return bestomega, bestpairs, bestwt
