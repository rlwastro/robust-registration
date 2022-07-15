"""Robust astrometric registration and cross-match of astronomical catalogs

This code does robust (Bayesian) cross-matches of catalogs with potentially
large astrometric errors.  The algorithm is described in Tian et al. (2019)
<https://ui.adsabs.harvard.edu/abs/2019AJ....158..191T>.

The xregistration module includes code that implements a catalog cross-match
with astrometric errors.  The algorithm uses a Bayesian approach to handle
objects that do not exist in both catalogs.  This version of the algorithm
implements the "ring" algorithm, which subsets all pairs within an initial
search radius R into overlapping rings.  This approach allows it to find
shifts that are much larger than the positional uncertainties in the catalogs.
It is particularly appropriate for catalogs from Hubble Space Telescope and
other small field telescopes that have potentially large astrometric errors.
The code in the xregistration/estimation.py module also uses a simple annealing
schedule for the astrometric uncertainty, the &sigma; value, to improve
convergence in the iteration.

The Jupyter notebook demonstrates using the robust registration algorithm to
cross-match catalogs with rotation and shift.  The first part of this notebook
tests the algorithm on simulated HST/ACS/WFC catalogs. The second part
demonstrates the cross-registration of a real HST image with a large shift (from
the HLA catalog) to the Gaia DR2 catalog of the same field.  We also compare the
robust estimation results with the results from the method of least-squares
(Budavári & Lubow 2012).

References

Tian, F., Budavári, T., Basu, A., Lubow, S.H., & White, R.L. (2019) Robust
Registration of Astronomy Catalogs with Applications to the Hubble Space
Telescope.
The Astronomical Journal, 158, 191. doi:10.3847/1538-3881/ab3f38
<https://ui.adsabs.harvard.edu/abs/2019AJ....158..191T>.

Budavári, T., & Lubow, S.H. (2012) Catalog Matching with Astrometric Correction
and its Application to the Hubble Legacy Archive.
The Astrophysical Journal, 761, 188. doi:10.1088/0004-637X/761/2/188
<https://ui.adsabs.harvard.edu/abs/2012ApJ...761..188B>
"""

from . import simulation, estimation, est_catalog

__all__ = ["simulation","estimation","est_catalog"]
