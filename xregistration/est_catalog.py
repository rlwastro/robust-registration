"""Utility functions

This module includes functions to read HLA and Gaia catalogs plus
some useful utility functions for converting between RA, Dec spherical
coordinates and 3-D Cartesian xyz coordinates.
"""

import numpy as np

from astropy.io import ascii, votable
from astropy.time import Time
import os, requests, sys, warnings
from io import StringIO

# global variables
# convert units from radians to degree
r2d = 180/np.pi
# convert units from degree to radians
d2r = np.pi/180

def getmultiwave(dataset, url='http://hla.stsci.edu/cgi-bin/getdata.cgi'):
    """Extract multiwave catalog from the Hubble Legacy Archive (HLA) for the given dataset.
    Typical dataset name is hst_10188_10_acs_wfc (with no filter). 
    
    :param dataset:    Image name.
    :param url:        URL of the HLA server to extract the image from.
    :type dataset:     str
    :type url:         str
    :returns:          (tab) astropy.Table: image table with (x,y) & (ra, dec) coordinates, magnitudes etc.
    """
    
    catname = dsname2total(dataset) + '_sexphot_trm.cat'
    r = requests.get(url, params={'format': 'csv', 'filename': catname})
    tab = ascii.read(r.text)
    # change column names to lowercase and delete useless _totmag columns
    for col in tab.colnames:
        lcol = col.lower()
        if lcol.endswith('_totmag'):
            del tab[col]
        elif lcol != col:
            tab.rename_column(col,lcol)
    # get the observation epoch and attach it as metadata
    tab.meta['epoch'] = getepoch(dataset)
    tab.meta['crval'] = getrefpos(dataset)
    tab.meta['dataset'] = dataset
    return tab

def dsname2total(dataset):
    """Convert dataset name to total image name.    
    Typical dataset name is hst_10188_10_acs_wfc (with no filter) but also works
    with hst_10188_10_acs_wfc_total.
    This translates Steve's WFPC2 dataset names (e.g. HST_08553_01_WFPC2_WFPC2) 
    to HLA-style names (hst_08553_01_wfpc2_total_wf).

    :param dataset:    Image name to look for
    :type dataset:     str
    :returns:          (totalname): string with total image name.
    """
        
    dataset = dataset.lower()
    # strip total off if it is already there
    if dataset.endswith('_total'):
        dataset = dataset[:-6]
    # convert dataset name to catalog name
    i = dataset.find('_wfpc2')
    if i >= 0:
        totalname = dataset[0:i+6] + '_total_wf'
    else:
        totalname = dataset + '_total'
    return totalname

def gaiaquery(ramin, decmin, ramax, decmax, version='dr2',
              url='http://hla.stsci.edu/cgi-bin/gaiaquery'):
    """
    Return Gaia catalog for the RA/Dec box. 
    
    :param ramin:      Minimum RA, units in degrees.
    :param decmin:     Minimum Dec, units in degrees.
    :param ramax:      Maximum RA, units in degrees.
    :param decmax:     Maximum Dec, units in degrees.
    :param version:    Version of the Gaia catalog. Either 'dr1' or 'dr2'.
    :param url:        Gaia server.
    :type ramin:       float
    :type decmin:      float
    :type ramax:       float
    :type decmax:      float
    :type version:     str
    :type url:         str
    :returns:          (tab) astropy.Table: Gaia catalog table with (ra,dec) coordinates, magnitudes etc.
    """

    bbox = [ramin, decmin, ramax, decmax]
    sbbox = ','.join([str(x) for x in [ramin, decmin, ramax, decmax]])
    vlist = ['dr1','dr2']
    if version not in ['dr1', 'dr2']:
        raise ValueError("version '{}' must be dr1 or dr2".format(version))
    r = requests.get(url, params={'bbox': sbbox, 'version':version, 
                                  'extra':'ra_dec_corr,phot_bp_mean_mag,phot_rp_mean_mag'})
    tab = ascii.read(r.text, data_start=2)
    # change column names to lowercase
    for col in tab.colnames:
        lcol = col.lower()
        if lcol != col:
            tab.rename_column(col,lcol)
    return tab

# global cache to speed repeated calls for info on data
hlacache = {}

def gethlainfo(dataset, url="http://hla.stsci.edu/cgi-bin/hlaSIAP.cgi", params=None):
    """Get info on the observation from the HLA SIAP server.
    
    Typical dataset name is hst_10188_10_acs_wfc (with no filter).
    This translates Steve's WFPC2 dataset names (e.g. HST_08553_01_WFPC2_WFPC2) 
    to HLA-style names (hst_08553_01_wfpc2_total_wf)
    
    :param dataset:    Image name.
    :param url:        URL of the HLA SIAP server to extract the image from.
    :param params:     Dictionary of extra parameters to include
    :type dataset:     str
    :type url:         str
    :type params:      dict
    :returns:          astropy.Table: image table with (x,y) & (ra, dec) coordinates, magnitudes etc.
    """
    
    totalname = dsname2total(dataset)
    try:
        return hlacache[totalname]
    except KeyError:
        pass
    if not params:
        params = {}
    params = dict(params, config='ops', pos='0,0', size='180', imagetype='combined',
                filter='detection', format='image/fits', visit=totalname)
    pstring = "&".join(["{}={}".format(x[0],x[1]) for x in params.items()])
    rurl = "{}?{}".format(url,pstring)
    # annoying way to get rid of progress message
    try:
        save_stdout = sys.stdout
        sys.stdout = StringIO()
        # suppress a bunch of irrelevant warnings while parsing the VOTable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vtab = votable.parse_single_table(rurl, pedantic=False)
    finally:
        sys.stdout = save_stdout
    tab = vtab.to_table(use_names_over_ids=True)
    # fix another annoyance by changing 'object' columns to strings
    for c in tab.colnames:
        if str(tab[c].dtype)=='object' and vtab.get_field_by_id_or_name(c).datatype == 'char':
            tab[c] = tab[c].astype(str)
    hlacache[totalname] = tab
    return tab

def getepoch(dataset):
    """Get the epoch for an HLA dataset.
    This uses the HLA SIAP server to get the info.

    :param dataset:    Image name.
    :type dataset:     str
    :returns:          float: date in decimal years.
    """
    
    tab = gethlainfo(dataset)
    date = Time(tab['StartTime'][0])
    return date.decimalyear

def getrefpos(dataset):
    """Return the reference position (crval1,crval2) for the HLA combined image.
    This uses the HLA SIAP server to get the info.
    
    :param dataset:    Image name.
    :type dataset:     str
    :returns:          tuple of floats.
    """
    
    tab = gethlainfo(dataset)
    return tuple(tab['crval'][0])

def radec2xyz(ra,dec):
    """Convert ra and dec arrays to xyz values
    ra, dec both may be scalars or arrays. 
    If both are arrays, they must have the same lengths.

    :param ra:      RA, in degrees. 
    :param dec:     Dec, in degrees.
    :type ra:       float or array 
    :type dec:      float or array
    :returns:       tuple (cxyz): [len(ra,dec),3], in radians. 
    """
    
    try:
        nra = len(ra)
        ra = np.asarray(ra)
    except TypeError:
        nra = 1
    try:
        ndec = len(dec)
        dec = np.asarray(dec)
    except TypeError:
        ndec = 1
    n = nra
    if n == 1:
        n = ndec
    elif ndec != nra and ndec != 1:
        raise ValueError("Mismatched array lengths for ra [{}], dec [{}]".format(nra,ndec))
    cxyz = np.zeros((n,3),dtype=np.float)
    rarad = d2r*ra
    decrad = d2r*dec
    cdec = np.cos(decrad)
    cxyz[:,0] = cdec*np.cos(rarad)
    cxyz[:,1] = cdec*np.sin(rarad)
    cxyz[:,2] = np.sin(decrad)
    return cxyz

def xyz2radec(cxyz):
    """Convert xyz value to RA and Dec arrays.
    
    :param cxyz:    Input (x,y,z) coordinates, in radians. 
    :type cxyz:     numpy.ndarray
    :returns:       (ra,dec) in degrees.
    """
    
    cxyz = np.asarray(cxyz)
    if len(cxyz.shape) == 1 and cxyz.shape[0] == 3:
        cxyz = cxyz.reshape((1,3))
    elif not (len(cxyz.shape) == 2 and cxyz.shape[1] == 3):
        raise ValueError("cxyz must be [3] or [n,3]")
    # normalize cxyz
    cxyz = cxyz / np.sqrt((cxyz**2).sum(axis=-1))[:,np.newaxis]
    dec = r2d*np.arcsin(cxyz[:,2])
    ra = r2d*np.arctan2(cxyz[:,1],cxyz[:,0])
    return (ra,dec)

def cat2xyz(cat, ra='ra', dec='dec'):
    """Return array [len(cat),3] with xyz values
    
    :param cat:    Input catalog with RA, DEC coordinates in degrees. 
    :param ra:     Select RA coordinates from the input catalog.
    :param dec:    Select Dec coordinates from the input catalog.
    :type cat:     astropy.Table
    :type ra:      str
    :type dec:     str
    :returns:      (xyz) tuple in radians.
    """
    
    return radec2xyz(cat[ra],cat[dec])
    
def getdeltas(ra0,dec0,ra1,dec1):
    """Compute shifts in arcsec between two positions.
    Input ra,dec units in degrees. At least one of ra0,dec0 or ra1,dec1 should be arrays
    
    :param ra0:    RA coordinates of catalog.
    :param dec0:   Dec coordinates of catalog.
    :param ra1:    RA coordinates of reference.
    :param dec1:   Dec coordinates of reference.
    :type ra0:     float or array
    :type dec0:    float or array
    :type ra1:     float or array
    :type dec1:    float or array
    :returns:      shifts dra, ddec in arcsec.

    """
    dra = ra1-ra0
    dra[dra > 180] -= 360
    dra[dra < -180] += 360
    dra = dra*np.cos(d2r*dec0)*3600
    ddec = (dec1-dec0)*3600
    return dra, ddec

def xyz2delta(dxyz,ra0,dec0):
    """Convert array of dxyz[*,3] values to dra,ddec given reference ra0,dec0.
    
    :param dxyz:   Input data array, (x,y,z) coordinates.
    :param ra0:    RA coordinates of reference catalog.
    :param dec0:   Dec coordinates of reference catalog.
    :type dxyz:    np.ndarray
    :type ra0:     float or array
    :type dec0:    float or array
    :returns:      shifts dra, ddec in arcsec.

    """
    if len(dxyz.shape) != 2 or dxyz.shape[1] != 3:
        raise ValueError("dxyz must be [*,3]")
    xyz0 = radec2xyz(ra0,dec0)
    xyz = dxyz + xyz0
    ra, dec = xyz2radec(xyz)
    dra, ddec = getdeltas(ra0,dec0,ra,dec)
    return dra, ddec
