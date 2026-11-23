import numpy as np
from scipy.stats import binned_statistic
import h5py
from nbodykit.lab import HDFCatalog
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def power_spectrum_2D(field1, field2, Lbox):
    """ Compute power spectrum 
    
    Args:
        field1 (array):
            First 2D field to measure power spectrum
        field2 (array):
            Second 2D field to measure power spectrum
        Lbox (float):
            Physical box length of 2D fields
    
    Returns:
        Pk (array): Power spectrum for different values of k

        ks (array): Wavenumber k values
    """
    Ngrid, _ = field1.shape
    dk = 2 * np.pi / Lbox
    kmax = np.pi / Lbox * Ngrid
    kbins = np.arange(0, kmax+dk/2, dk)
    ks = 0.5 * (kbins[:-1] + kbins[1:])
    kgrid = 2 * np.pi * np.fft.fftfreq(Ngrid, Lbox/Ngrid)
    kgrid = np.meshgrid(kgrid, kgrid, indexing='ij')
    kgrid = np.sqrt(kgrid[0]**2 + kgrid[1]**2)

    f1 = np.fft.fft2(field1) * Lbox / Ngrid**2
    f2 = np.fft.fft2(field2) * Lbox / Ngrid**2
    power = np.real(f1 * f2.conj())
    
    Pk, _, _ = binned_statistic(kgrid.flatten(), power.flatten(), bins=kbins)
    Pk[0] = 0
    ks[0] = 0
    return Pk, ks

class SimConfig():
    """ Simulation Config class.
    
    Args:
        sim_dir (str): Directory name of simulation output (NOT FULL PATH)
        sim_inds (int or tuple or list): Indices of simulations if multiple \
            simulations, int for number of sims, tuple for index range, list \
            for list of indices
        snap_inds (int or tuple or list): Indices of snapshots, int for number \
        of snapshots, tuple of index range, list for list of indices
        use_ics (bool): Indicates whether first snapshot is from ics file
    """
    def __init__(self, sim_dir, sim_inds, snap_inds, use_ics=True):
        # Simulation indices
        if type(sim_inds) is int:
            sim_inds = list(range(sim_inds))
        elif type(sim_inds) is tuple and len(sim_inds) > 1:
            sim_inds = list(range(sim_inds[0], sim_inds[1]))
        self.sim_inds = sim_inds
        
        if len(self.sim_inds) > 1:
            self.sim_dirs = [sim_dir + f'_{s:02d}' for s in self.sim_inds]
        else:
            self.sim_dirs = [sim_dir]

        # Snapshot indices
        if type(snap_inds) is int:
            snap_inds = list(range(snap_inds))
        elif type(snap_inds) is tuple and len(snap_inds) > 1:
            snap_inds = list(range(snap_inds[0], snap_inds[1]))
        self.snap_inds = snap_inds
        self.n_sims = len(self.sim_inds)
        self.n_snaps = len(self.snap_inds)
        self.shape = (self.n_sims, self.n_snaps)

        self.use_ics = use_ics
    
    def get_config(self):
        return self.sim_dirs, self.sim_inds, self.snap_inds

class GadgetDataset(Dataset):
    """ Dataset object to load simulations. 
    
    Args:
        Ngrid (int): Size of grid in pixels
        Lbox (float): Physical size of grid in Mpc/h
        root_dir (str): Root directory of all simulation outputs
        sim_configs (dict[str,SimConfig]): Dictionary of simulation configs
    """
    def __init__(self, 
        Ngrid: int,
        Lbox: float,
        root_dir: str,
        sim_configs: dict,
        **kwargs
    ):
        self.root = Path(root_dir)
        self.Ngrid = Ngrid
        self.Lbox = Lbox
        self.sim_configs = sim_configs
        self.kwargs = kwargs

        self.n_sims = 0
        self.n_snaps = 0
        for sim_type, sim_config in sim_configs.items():
            n_sims, n_snaps = sim_config.shape
            self.n_sims += n_sims
            if self.n_snaps == 0:
                self.n_snaps = n_snaps
            else:
                assert self.n_snaps == n_snaps, "All simulations must use the same number of snapshots"

    def __len__(self):
        return self.n_sims * self.n_snaps
    
    def __getitem__(self, idx):
        s_i, ss_i = np.unravel_index(idx, (self.n_sims, self.n_snaps))
        sim_config, s_i = self.get_indexed_config(s_i)
        ss_i = sim_config.snap_inds[ss_i]

        inpath: Path = self.root / sim_config.sim_dirs[s_i] 
        if ss_i == 0 and sim_config.use_ics:
            inpath /= f'snapshot_ics_{ss_i:03d}.hdf5'
        else:
            inpath /= f'snapshot_{ss_i:03d}.hdf5'            
        with h5py.File(inpath, 'r') as f:
            header = f['Header'].attrs
            z = header['Redshift']

        cat = HDFCatalog(str(inpath))

        mesh = cat.to_mesh(
            position='PartType1/Coordinates',
            Nmesh=self.Ngrid,
            BoxSize=self.Lbox,
            interlaced=self.kwargs.get('interlaced', True),
            compensated=self.kwargs.get('compensated', True),
            resampler=self.kwargs.get('resampler', 'cic')
        )
        return np.array(mesh.compute()), z, s_i, ss_i

    def get_indexed_config(self, idx):
        cur_idx = idx
        for sim_type, sim_config in self.sim_configs.items():
            if cur_idx < sim_config.n_sims:
                return sim_config, cur_idx
            cur_idx -= sim_config.n_sims
        raise IndexError('Index out of range')

def make_projected_fields(dataset: GadgetDataset):
    """ Make projected density fields from dataset"""
    n_sims = dataset.n_sims
    n_snaps = dataset.n_snaps
    Ngrid = dataset.Ngrid
    dataloader = DataLoader(dataset, batch_size=n_snaps)

    projected_fields = np.zeros((n_sims, n_snaps, 3, Ngrid, Ngrid))
    redshifts = np.zeros(n_snaps)
    print('Making projected fields from simulations...')
    for i, (mesh3d, z, s_i, ss_i) in enumerate(dataloader):
        print(f' Simulation {i}')
        for p in range(3):
            projected_fields[i,:,p,:,:] = np.mean(mesh3d.numpy() - 1, axis=p+1)
        if i == 0:
            redshifts = z
    
    return projected_fields, redshifts.numpy()