"""Functions to generate simulated catalogs used for testing"""

import numpy as np
import pandas as pd

# global variables
# convert units from arcseconds to radians
arc2rad = 3600 * 180 / np.pi

def mock(num, arcsec, seed=None):
    """Generates mock objects with their true xyz directions near (0,0,1) and a faked physical property u
    
    :param num:    Specify the number of objects in the mock universe.
    :param arcsec: Specify the size of the mock image, units in arcseconds.
    :param seed:   Initialize a random seed for simulating from a standard normal distribution, default is None.
    :type num:     int
    :type arcse:   float
    :type seed:    int or None
    :returns:      (df) pandas.DataFrame with 4 columns. The first 3 columns are the xyz coordinates of the simulated 
                   mock objects. The last column u represents the random properties assigned to each of the object.
    """

    if seed is not None: np.random.seed(seed)
    obj = np.random.random(size=(num,4))
    obj[:,:2] *= arcsec / arc2rad # xy scaling: convert to radians
    obj[:, 2] = np.sqrt(1 - np.square(obj[:,0]) - np.square(obj[:,1])) # calculate z
    df = pd.DataFrame(obj, columns=list('xyzu'))
    return df

def cat(mdf, sigma, umin=0, umax=1, seed=None):
    """
    Generate a catalog from a mock DataFrame.
    Subset of sources based on (umin,umax).
    Add small random perturbation to directions based on sigma. 
    
    :param mdf:    Input mock universe dataframe.
    :param sigma:  Specify an uncertainty parameter for the simulated cataolog perturbation, units in arcseconds
    :param umin:   Minimum for subseting sources based on property u.
    :param umax:   Maximum for subseting sources based on property u.
    :param seed:   Initialize a random seed for simulating from a normal distribution with 
                   standard deviation of sigma(in radians). Default is None.
    :type mdf:     pandas.DataFrame
    :type sigma:   float
    :type umin:    float
    :type umax:    float
    :type seed:    int or None
    :returns:      (df) pandas.DataFrame with 5 columns. The first 3 columns are the xyz coordinates of the simulated 
                   catalogs. The 4th column u represents the random properties assigned to each of source, 
                   same as the input mock. The last column is a list of boolean values, indicating if the 
                   source is selected based on input selection interval (umin, umax).
    """
    if seed is not None: np.random.seed(seed)
    sigrad = sigma / arc2rad #convert units to radians
    r = mdf[list('xyz')].values + np.random.normal(scale=sigrad, size=(mdf.shape[0],3)) # plus random perturbation
    r /= np.sqrt(np.square(r).sum(axis=1))[:,np.newaxis] # normalize
    df = pd.DataFrame(r, columns=list('xyz'))
    df['u'] = mdf.u
    df['Selected'] = np.logical_and(mdf.u > umin, mdf.u < umax)
    return df

def trf(cat, omega):
    """Apply shift and rotation as specified by the 3D transformation based on omega.
    
    :param cat:    Input catalog dataframe.
    :param omega:  3D transformation vector to be applied to the input catalog.
    :type cat:     pandas.DataFrame
    :type omega:   float
    :returns:      (df) pandas.DataFrame with 4 columns. The first 3 columns are the xyz coordinates of the 
                   transformed catalogs. The last column is a list of boolean values, indicating if the 
                   source is selected, same as the input catalog.
    """
    r = cat[list('xyz')].values 
    r += np.cross(omega, r)
    df = pd.DataFrame(r, columns=list('xyz'))
    df.index = cat.index
    if 'Selected' in cat:
        df['Selected'] = cat.Selected
    return df

def randomega(cat, scale=1, seed=None):
    """Apply random shift and rotation to a simulated catalog
    Returns the random rotation and the transformed catalog
    
    :param cat:    Input catalog dataframe.
    :param scale:  To create the 3D transformation vector from a random normal distribution, specify the standard deviation.
    :param seed:   Initialize a random seed for simulating from a normal distribution, default is None.
    :type cat:     pandas.DataFrame
    :type scale:   float
    :type seed:    int or None
    :returns:      (omega, df): the random omega vector generated, and a pandas.DataFrame as the transformed catalog.
    """
    if seed is not None: np.random.seed(seed)
    scalerad = scale /arc2rad # convert units from arcsec to radians
    omega = np.random.normal(scale=scalerad, size=3) 
    df = trf(cat, omega)
    return omega, df

def true_pairs(cat, ref):
    """obtain list of true pairs from simulated catalogs.
    
    :param cat:    Input catalog dataframe to be corrected, with (x,y,z) coordinates.
    :param ref:    Reference catalog dataframe with (x,y,z) coordinates. 
    :type cat:     pd.DataFrame      
    :type ref:     pd.DataFrame
    :returns:      DataFrame of concatenated catalogs (cat,ref).
    """
    catalogs = [cat,ref]
    return(pd.concat([df[list('xyz')] for df in catalogs], axis=1).dropna().values)

def getsep(cat, ref, value="max"):
    """maximum separation betwen true pairs of the simulated catalogs.
    
    :param cat:    Input catalog dataframe to be corrected, with (x,y,z) coordinates.
    :param ref:    Reference catalog dataframe with (x,y,z) coordinates. 
    :param value:  Result, choose from "max" or "mean".
    :type cat:     pd.DataFrame      
    :type ref:     pd.DataFrame
    :type value:   str
    :returns:      maximum or average separation between true pairs, units in arcseconds.
    """
   
    tp = true_pairs(cat,ref)
    if value == "max":
        sep = np.sqrt(np.square(tp[:,:3]-tp[:,3:]).sum(axis=1).max()) * arc2rad
    if value == "mean":
        sep = np.mean(np.sqrt(np.square(tp[:,:3]-tp[:,3:]).sum(axis=1))) * arc2rad
    return(sep)

