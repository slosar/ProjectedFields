import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.fft as fft
from scipy.ndimage import map_coordinates
import dask
import dask.bag as db
from nbodykit.lab import HDFCatalog, ArrayCatalog
from nbodykit.cosmology import Cosmology
# from projectedfields import *

def evolve_lpt2d(cosmo: Cosmology, field_init, Ngrid, Lbox, z, z_ini=63.0, lpt_order=1, return_catalog=False, **kwargs):
    """ Evolve intitial conditions using LPT-2D. 
    
    Args:
        cosmo (nbodykit.cosmology.Cosmology): nbodykit Cosmology object
        field_init (numpy.ndarray): 2D initial condition density field
        Ngrid (int): Size of grid in pixels
        Lbox (float): Physical size of grid in Mpc/h
        z (float): Redshift to evolve to
        z_ini (float): Initial redshift
        lpt_order (int): Order of LPT evolution (only supports up to 2nd order)
        return_catalog (bool): Set True to return nbodykit Catalog object
        kwargs (dict): Set optional kwargs to be passed into nbodykit Catalog.to_mesh()
            ['interlaced', 'compensated', 'resampler']
    """
    # Get displacements 
    f = np.fft.fft2(field_init)
    kf = fft.fftfreq(Ngrid, Lbox/Ngrid) * 2 * np.pi
    kgrid = np.array(np.meshgrid(kf, kf, indexing='ij'))
    kmag2 = kgrid[0]**2 + kgrid[1]**2
    kmag2[0,0] = np.nan
    disp_f = 1j * kgrid / kmag2 * f
    disp_f[:,0,0] = 0
    gfactor = cosmo.scale_independent_growth_factor(z) \
            / cosmo.scale_independent_growth_factor(z_ini)
    disp = gfactor * fft.ifft2(disp_f).real
    # 2nd-order LPT
    if lpt_order > 1:
        kgrid2 = kgrid * kgrid[:,None]
        phi_dd_f = -kgrid2 / kmag2 * f
        phi_dd_f[:,:,0,0] = 0
        phi_dd = np.fft.ifft2(phi_dd_f)
        phi2 = phi_dd[0,0] * phi_dd[1,1] - phi_dd[0,1]**2
        phi2_f = np.fft.fft2(phi2)
        disp2_f = 1j * kgrid / kmag2 * phi2_f
        disp2_f[:,0,0] = 0
        disp2 = 3/7 * gfactor**2 * fft.ifft2(disp2_f).real
        disp += disp2

    # Set up initial particle grid (2D)
    x = np.linspace(0, Lbox, Ngrid, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    pos0 = np.stack([xx.ravel(), yy.ravel(), np.zeros(Ngrid**2)], axis=1)

    # Update positions
    grid_pos = (pos0 / (Lbox / Ngrid)).T[:2]
    disp_interp = [map_coordinates(disp[i], grid_pos, mode='wrap') for i in range(2)]
    disp_interp.append(np.zeros_like(disp_interp[0]))
    disp_interp = dask.array.array(disp_interp)
    pos = pos0 + disp_interp.T
    pos = pos % Lbox

    # Convert positions to mesh field
    data = {'Position': pos}
    cat = ArrayCatalog(data)
    mesh = cat.to_mesh(
        Nmesh=Ngrid,
        BoxSize=Lbox,
        interlaced=kwargs.get('interlaced', True),
        compensated=kwargs.get('compensated', True),
        resampler=kwargs.get('resampler', 'cic')
    )
    field = mesh.compute()
    field = np.mean(field-1, axis=2)

    if return_catalog:
        return field, cat

    return field