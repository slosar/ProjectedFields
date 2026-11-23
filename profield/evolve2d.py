import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.fft as fft
import dask
import dask.bag as db
from nbodykit.lab import HDFCatalog
from nbodykit.cosmology import Cosmology
from .projectedfields import *
import pyccl as ccl

OMEGA_MATTER = 0.308
OMEGA_LAMBDA = 0.692
OMEGA_BARYON = 0.0482
N = 128
L = 250

cosmo = Cosmology(h=0.678, Omega0_b=0.0482, Omega0_cdm=0.308, n_s=1.0)
cosmo.match(0.9)

def approx_linear_growth(z):
    a = 1 / (1+z)
    return a / (OMEGA_MATTER**(4/7) - OMEGA_LAMBDA + (1+OMEGA_MATTER/2)*(1+OMEGA_LAMBDA/70))

def approx_growth_factor(z, z0=0.0):
    return approx_linear_growth(z) / approx_linear_growth(z0)

def hubble(a):
    return np.sqrt(OMEGA_MATTER / a**3 + (1 - OMEGA_MATTER - OMEGA_LAMBDA) / a**2 + OMEGA_LAMBDA)

def growth_integrand(a):
    return 1 / (a * hubble(a))**3

def linear_growth(z):
    a = 1 / (1+z)
    result = integrate.quad(growth_integrand, 0, a)
    return hubble(a) * result[0]

def growth_factor(z, z0=63.0):
    return cosmo.scale_independent_growth_factor(z) / cosmo.scale_independent_growth_factor(z0)

def growth_factor2(z, z0=63.0):
    return linear_growth(z)**2 / linear_growth(z0)**2

def get_displacement(field, z, order=2):
    field_fft = fft.fftn(field)
    disp1 = np.zeros((2, *field_fft.shape), dtype=np.complex128)
    disp2 = np.zeros((2, *field_fft.shape), dtype=np.complex128)
    d2phi_dx2 = np.zeros(field_fft.shape, dtype=np.complex128)
    d2phi_dy2 = np.zeros(field_fft.shape, dtype=np.complex128)
    d2phi_dxdy = np.zeros(field_fft.shape, dtype=np.complex128)
    phi2 = np.zeros(field_fft.shape, dtype=np.complex128)
    kf = fft.fftfreq(N, L/N)
    test = 0
    for i, kx in enumerate(kf):
        for j, ky in enumerate(kf):
            if kx == 0 or ky == 0: continue
            kmag2 = kx**2 + ky**2
            disp1[0,i,j] = 1j * kx / kmag2 * field_fft[i,j]
            disp1[1,i,j] = 1j * ky / kmag2 * field_fft[i,j]
            d2phi_dx2[i,j] = kx**2 / kmag2 * field_fft[i,j]
            d2phi_dy2[i,j] = ky**2 / kmag2 * field_fft[i,j]
            d2phi_dxdy[i,j] = kx * ky / kmag2 * field_fft[i,j]
            # phi2[i,j] += fft.ifftn(kx**2 / kmag2 * field_fft[i,j]) * fft.ifftn(ky**2 / kmag2 * field_fft[i,j])
            # phi2[i,j] -= fft.ifftn(kx * ky / kmag2 * field_fft[i,j])**2
            # if i == 40 and j == 60: 
            #     print(i, j, phi2[i,j])
            #     print(fft.ifftn(kx**2 / kmag2 * field_fft[i,j]))
            #     print(fft.ifftn(ky**2 / kmag2 * field_fft[i,j]))
            #     print(fft.ifftn(kx*ky / kmag2 * field_fft[i,j])**2)
            #     test=1
            # disp2[0,i,j] = 1j * kx / kmag2 * fft.fftn(phi2[i,j])
            # disp2[1,i,j] = 1j * ky / kmag2 * fft.fftn(phi2[i,j])
    phi2 = fft.ifftn(d2phi_dx2) * fft.ifftn(d2phi_dy2) - fft.ifftn(d2phi_dxdy)**2
    # print(phi2.real)
    phi2_fft = fft.fftn(phi2)
    for i, kx in enumerate(kf):
        for j, ky in enumerate(kf):
            if kx == 0 or ky == 0: continue
            kmag2 = kx**2 + ky**2
            disp2[0,i,j] = 1j * kx / kmag2 * phi2_fft[i,j]
            disp2[1,i,j] = 1j * ky / kmag2 * phi2_fft[i,j]

    disp = np.zeros((2, *field_fft.shape))
    for d in range(2):
        disp1[d] = fft.ifftn(disp1[d])
        disp2[d] = fft.ifftn(disp2[d])
        disp[d] = (growth_factor(z)) * disp1[d].real 
        if order==2:
            disp[d] -= 3/7*growth_factor(z)**2 * disp2[d].real

    return disp

def update_pos(pos, disp):
    grid_pos = pos // (L / N)
    if grid_pos[0] >= 128: grid_pos[0] = 0
    if grid_pos[1] >= 128: grid_pos[1] = 0
    x = pos[0] + disp[0, int(grid_pos[0]), int(grid_pos[1])]
    y = pos[1] + disp[1, int(grid_pos[0]), int(grid_pos[1])]
    z = pos[2]
    return x, y, z

@dask.delayed
def f_update_pos(positions, disp):
    final_pos = []
    for pos in positions:
        final_pos.append(update_pos(pos, disp))
    return final_pos

def get_evolved_field(field, z, order=1):
    # cat = HDFCatalog('../../gadget4/examples/projectedfield/output/snapshot_ics_000.hdf5')
    cat = HDFCatalog('/home/khong/bnl/test/output_onlyPF/snapshot_ics_000.hdf5')
    positions, velocities = cat.read(columns=['PartType1/Coordinates','PartType1/Velocities'])
    print(f'Growth factor = {growth_factor(z)}')

    print('Computing displacements...')
    disp = get_displacement(field, z, order=order)

    print('Updating positions...')
    results = []
    for i in range(0, positions.shape[0], N**2):
        results.append(f_update_pos(positions[i:i+N**2], disp))
    results = dask.compute(*results)
    arr = dask.array.array(results)
    arr = np.reshape(arr, (N**3, 3))
    cat['PartType1/Coordinates'] = arr

    print('Computing mesh...')
    mesh = cat.to_mesh(
        position='PartType1/Coordinates',
        Nmesh=N,
        BoxSize=L,
        interlaced=True,
        compensated=True
    )
    mesh_arr = mesh.compute()
    field = np.sum(mesh_arr-1, axis=2)
    return field

if __name__=='__main__':
    name = 'test1'
    order = 2
    print('Loading initial fields')
    pfp = ProjectedFieldsPlotter(
        gadget_dir='',
        output_dir='output_onlyPF',
        inputs={},
        plot_dir='data/', 
        Nmesh=128, 
        load_prefix='100sims_N128_allsnaps'
    )
    # pfp = ProjectedFieldsPlotter(
    #     gadget_dir='',
    #     output_dir='output_onlyPF',
    #     inputs={},
    #     plot_dir='../plots/pf1', 
    #     Nmesh=128, 
    #     load_prefix='100sims_N128_allsnaps'
    # )

    field_init = pfp.fields[0,0,2] / 128 / (2*np.pi)
    for ss in range(len(pfp.redshifts)):
    # for ss in range(0, 2):
        ss=11
        z = pfp.redshifts[ss]
        print(f'z = {z:2.1f}')
        field_evolved = get_evolved_field(field_init, z, order=order)

        fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        im1 = ax[0].imshow(field_evolved)
        im2 = ax[1].imshow(pfp.fields[0,ss,2])#, clim=im1.get_clim())
        im1.set_clim(im2.get_clim())
        fig.savefig(f'/home/khong/ProField/plots/test/fields_{name}_order{order}_{ss:03}.png')

        k, pk = compute_power_spectrum(field_evolved, N, L)

        fig_ps, ax_ps = plt.subplots(1, 1, figsize=(4,4))
        ax_ps.loglog(k, pk, label='1LPT')
        ax_ps.loglog(k, pfp.Pks[0,ss,2], label='Simulated', color='k')
        ax_ps.loglog(k, pfp.mean_Pks[ss,2], label='Mean', color='tab:purple')
        # ax_ps.loglog(k, pfp_only.Pks[0,ss,2], label='OPM')
        ax_ps.legend()
        fig_ps.savefig(f'/home/khong/ProField/plots/test/powerspec_{name}_order{order}_{ss:03}.png')
        break
    # plt.show()

