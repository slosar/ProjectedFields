import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import nbodykit.lab as nbk
from nbodykit import setup_logging
from nbodykit.source.catalog import HDFCatalog
import sys
import os
import configparser
from ast import literal_eval
import scipy.fft as fft
import scipy.stats as stats
import multiprocessing as mp

class ProjectedFieldsPlotter():
    '''Class to plot N-body simulations from Gadget-4 using nbodykit.

    Parameters
    ----------
    gadget_dir : string
        Path to directory of Gadget-4 simulation.
    output_dir : string
        Base path from `gadget_dir` to directories containing output snapshots.
        If `nsims > 0`, then `output_dir` will be suffixed with sim number i.e. 'output_00'
        If `nsims = 0`, then use `output_dir` as output directory.
    nsims : int or list of ints
        Number of simulations to plot. If list, use specified indices.
    snapshots : int or list of ints
        Number of snapshots for each simulation. If list, use specified indices.
    plot_dir : string
        Output path for plots.
    '''
    _axes1 = {'z':(0,1), 'x':(1,2), 'y':(0,2)}
    _axes2 = {'z':2, 'x':0, 'y':1}
    
    def __init__(self, gadget_dir, output_dir, inputs, Nmesh=128, BoxSize=250,
                 plot_dir='../plots/', load_prefix=''):
        self.gadget_dir = gadget_dir
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        self.Nmesh = Nmesh
        self.BoxSize = BoxSize
        
        if not load_prefix:
            sims_list = []
            snapshots_list = []
            do_ics_list = []
            self.nsims = 0
            if type(inputs) is not list: inputs = [inputs]
            for i, input in enumerate(inputs):
                try: sims = input['sims']
                except: sims = 0
                try: snapshots = input['snapshots']
                except: snapshots = 1
                try: do_ics = input['do_ics']
                except: do_ics = False
                do_ics_list.append(do_ics)

                if type(sims) is int:
                    sims = list(range(sims))
                elif type(sims) is tuple:
                    sims = list(range(sims[0], sims[1]))
                sims_list.append(sims)
                if len(sims) == 0: self.nsims += 1
                else: self.nsims += len(sims)

                if type(snapshots) is int:
                    snapshots = list(range(snapshots))
                elif type(snapshots) is tuple:
                    snapshots = list(range(snapshots[0], snapshots[1]))
                if i > 0 and len(snapshots_list[i-1]) != len(snapshots):
                    raise ValueError('All simulations must contain same number of snapshots.')
                snapshots_list.append(snapshots)
            
            self.nsnaps = len(snapshots_list[0])
            self.fields = np.zeros(shape=(self.nsims, self.nsnaps, 3, self.Nmesh, self.Nmesh))
            self.redshifts = np.zeros(self.nsnaps)
            offset = 0
            print(f'Loading {self.nsims} simulations at redshifts {self.redshifts}...')
            for sims, snapshots, do_ics in zip(sims_list, snapshots_list, do_ics_list):
                for s_i, ss_i, redshift, mesh in self.iter_simulations(sims, snapshots, do_ics):
                    for p in self._axes2.values():
                        self.fields[s_i+offset, ss_i, p] = np.mean(mesh-1, axis=p)
                    if s_i == 0:
                        self.redshifts[ss_i] = redshift
                offset += len(sims)
        else:
            prefix = load_prefix
            prefix = '_'+prefix if prefix else prefix
            data = np.load(self.plot_dir+'/'+self.output_dir+f'_fields{prefix}.npz')
            print('Loading from file: ', self.plot_dir+'/'+self.output_dir+f'_fields{prefix}.npz')
            self.fields = data['fields']
            self.redshifts = data['redshifts']
            self.Nmesh = self.fields.shape[-1]
            self.nsims = self.fields.shape[0]
            self.nsnaps = self.fields.shape[1]

        self.mean_fields = None
        self.resid_fields = None
        self.var_fields = None
        
        self.k = np.zeros(shape=self.Nmesh//2)
        self.Pks = np.zeros(shape=(self.nsims, self.nsnaps, 3, self.Nmesh//2))
        self.mean_Pks = np.zeros(shape=(self.nsnaps, 3, self.Nmesh//2))
        self.resid_Pks = np.zeros(shape=(self.nsims, self.nsnaps, 3, self.Nmesh//2))
        self.var_Pks = np.zeros(shape=(self.nsnaps, 3, self.Nmesh//2))

        self.compute_statistic_field()
        self.compute_power_spectra()

    def get_mesh_projections(self, sim_ind):
        s_i, ss_i = sim_ind
        if ss_i==0: print(f'== Adding Simulation {self.nsims[s_i]:02} ==')
        print(f'---- Adding snapshot {self.snapshots[ss_i]:03} ----')
        output_dir = self.output_dir_list[s_i]
        ss = self.snapshots[ss_i]
        inpath = self.gadget_dir+'/'+output_dir+f'/snapshot_{ss:03}.hdf5'
        if self.ics and ss_i==0:
            print('---- (using IC snapshot) ----')
            inpath = self.gadget_dir+'/'+output_dir+'/snapshot_ics_000.hdf5'
        cat = HDFCatalog(inpath)
        mesh = cat.to_mesh(position='PartType1/Coordinates', Nmesh=int(np.cbrt(cat.size)), BoxSize=self.BoxSize)
        print('TEST')
        mesh = mesh.preview(Nmesh=self.Nmesh)
        print(mesh.shape)
        meshes = np.array([np.sum(mesh, axis=p) for p in range(3)])
        print('test')
        return meshes

    def load_simulations_parallel(self, nprocs, inpaths):
        print(f'Loading simulations in parallel ({nprocs} processes)')
        with cf.ProcessPoolExecutor(nprocs) as p:
            futures = {p.submit(self.get_mesh_projections, inpath) for inpath in inpaths}
            for future in cf.as_completed(futures):
                yield future.result()

    def iter_simulations(self, sims, snapshots, do_ics=False):
        if len(sims) == 0: sims = [-1]
        for s_i, s in enumerate(sims):
        # for s_i, (s, output_dir) in enumerate(zip(self.nsims, self.output_dir_list)):
            if s == -1: output_dir = self.output_dir
            else: output_dir = self.output_dir+f'_{s:02}'
            inpath_base = self.gadget_dir+'/'+output_dir
            print(f'== Adding Simulation from directory: {output_dir} ==')
            for ss_i, ss in enumerate(snapshots):
                print(f'---- Adding snapshot {ss} ----')
                inpath = inpath_base+f'/snapshot_{ss:03}.hdf5'
                if do_ics and ss_i==0:
                    print('----     (using IC snapshot)     ----')
                    inpath = inpath_base+'/snapshot_ics_000.hdf5'
                cat = HDFCatalog(inpath)

                with h5py.File(inpath, 'r') as f:
                    header = f['Header'].attrs
                    redshift = header['Redshift']
                
                mesh = cat.to_mesh(
                    position='PartType1/Coordinates', 
                    Nmesh=int(np.cbrt(cat.size)), 
                    BoxSize=self.BoxSize,
                    interlaced=False,
                    compensated=False,
                    resampler='tsc',
                )
                yield s_i, ss_i, redshift, mesh.compute()

    def plot(self, sim_idx=(0,0), figax=None, plot_type='PF', statistic=None, projection='z', **kwargs):
        if figax is None:
            fig, ax = plt.subplots(1,1)
        else: fig, ax = figax
        try: ax = ax[0]
        except: pass

        s, ss = sim_idx
        if s >= self.nsims or ss >= self.nsnaps:
            print('Invalid simulation index ', sim_idx)
            print('Skipping...')
            return fig, ax
        
        try: assert plot_type in ['PF', 'PPS']
        except: raise ValueError("Invalid plot type... must be either 'PF' or 'PPS'.")
        try: assert projection in self._axes2.keys()
        except: raise ValueError("Projection axis must be 'x', 'y', or 'z'.")
        p = self._axes2[projection]

        if statistic is None:
            mesh = self.fields[s,ss,p]
            Pk = self.Pks[s,ss,p]
        elif statistic=='mean':
            try: assert self.mean_fields is not None
            except: raise ValueError('Please compute average field first.')
            mesh = self.mean_fields[ss,p]
            Pk = self.mean_Pks[ss,p]
        elif statistic=='residual':
            try: assert self.resid_fields is not None
            except: raise ValueError('Please compute residual fields first.')
            mesh = self.resid_fields[s,ss,p]
            Pk = self.resid_Pks[s,ss,p]
        elif statistic=='variance':
            try: assert self.resid_fields is not None
            except: raise ValueError('Please compute variance fields first.')
            mesh = self.var_fields[ss,p]
            Pk = self.var_Pks[ss,p]
        else: raise ValueError("Invalid statistic... must be either None, 'mean', 'residual', or 'variance'.")

        if plot_type=='PF':
            vmin = None if 'vmin' not in kwargs.keys() else kwargs['vmin']
            vmax = None if 'vmax' not in kwargs.keys() else kwargs['vmax']
            ax.imshow(mesh, extent=[0, self.BoxSize, 0, self.BoxSize], vmin=vmin, vmax=vmax)
        elif plot_type=='PPS':
            k = self.k
            # k, Pk = compute_power_spectrum(mesh, Nmesh=self.Nmesh, BoxSize=self.BoxSize)
            ax.loglog(k, Pk, **kwargs)
        return fig, ax

    def plot_projections(self, sim_idx=(0,0), figax=None, plot_type='PF', statistic=None, do_title=False, **kwargs):        
        # Set max size (3 rows)
        if figax is None:
            fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True)
            do_title=True
        else:
            fig, ax = figax
            if len(ax.shape)==2: ax = ax[0]
            try: assert ax.size==3
            except: raise ValueError('Axes object must have 3 columns.')

        for p_i, p in enumerate(self._axes1.keys()):
            curax = ax[p_i]
            fig, curax = self.plot(
                sim_idx=sim_idx,
                figax=(fig, curax),
                plot_type=plot_type,
                statistic=statistic,
                projection=p,
                **kwargs
            )
            if do_title: self.set_projection_labels((fig, ax))
        return fig, ax
    
    # def plot_histogram(self, sim_idx=(0,0), figax=None, )

    def plot_all(self, sim_idx=(None,0), figax=None, plot_type='PPS', statistic=None, projection='z', **kwargs):
        if figax is None:
            fig, ax = plt.subplots(1,1)
        else: fig, ax = figax
        try: ax = ax[0]
        except: pass

        if sim_idx[0] is None: plot_sims = self.nsims
        else: plot_sims = self.nsims[:sim_idx[0]]

        for s_i, s in enumerate(plot_sims):
            fig, ax = self.plot(
                sim_idx=(s_i,sim_idx[1]),
                figax=(fig, ax),
                plot_type=plot_type,
                statistic=statistic,
                projection=projection,
                **kwargs
            )

    def set_axes_labels(self, ax, x=True, y=True):
        if any(isinstance(x, mpl.image.AxesImage) for x in ax.get_children()):
            if x: ax.set_xlabel(r'$\rm h^{-1} Mpc$')
            if y: ax.set_ylabel(r'$\rm h^{-1} Mpc$')
        elif any(isinstance(x, mpl.lines.Line2D) for x in ax.get_children()):
            if x: ax.set_xlabel(r'$k [\rm h\ Mpc^{-1}]$')
            if y: ax.set_xlabel(r'$P(k) [\rm h^{-3} Mpc^3]$')
        else: raise ValueError('Axes object does not contain a line (PPS) or image (PF)')
    
    def compute_statistic_field(self, statistic='all'):
        if statistic=='all': statistic = 'mean_residual_variance'
        # stats = ['mean', 'residual', 'variance']
        if 'mean' in statistic:
            print('== Computing mean field ==')
            self.mean_fields = np.mean(self.fields, axis=0)
        if 'residual' in statistic:
            print('== Computing residual fields ==')
            self.resid_fields = self.fields - self.mean_fields
        if 'variance' in statistic:
            print('== Computing variance field ==')
            self.var_fields = np.sum(self.resid_fields**2, axis=0) / self.nsims

    def compute_power_spectra(self):
        print('== Computing power spectra ==')
        i = 0
        fields = [self.fields, self.mean_fields, self.resid_fields, self.var_fields]
        Pks = [self.Pks, self.mean_Pks, self.resid_Pks, self.var_Pks]
        for field, Pk in zip(fields, Pks):
            if field is None: continue
            for ss_i in range(self.nsnaps):
                for p in self._axes2.values():
                    if len(field.shape)==4:
                        mesh = field[ss_i,p]
                        k, Pk[ss_i,p] = compute_power_spectrum(mesh, Nmesh=self.Nmesh, BoxSize=self.BoxSize)
                        if i==0:
                            self.k = k
                            i = 1
                    else:
                        for s_i in range(self.nsims):
                            mesh = field[s_i,ss_i,p]
                            k, Pk[s_i,ss_i,p] = compute_power_spectrum(mesh, Nmesh=self.Nmesh, BoxSize=self.BoxSize)

    def set_projection_labels(self, figax):
        fig, ax = figax
        if len(ax.shape)==2: ax = ax[0]
        for p_i, p in enumerate(self._axes1.keys()):
            ax[p_i].set_title(f'{p}-axis projection')

    def save_fields(self, prefix=''):
        prefix = '_'+prefix if prefix else prefix
        filename = self.plot_dir+'/'+self.output_dir+f'_fields{prefix}.npz'
        print('>>> Saving fields to: ', filename)
        np.savez(filename, 
            fields=self.fields, 
            redshifts=self.redshifts,
        )

def plot_custom(plot_name, config, pfp_normal, pfp_onlyPF=None, pfp_zeroPF=None, save=True): 
    ''' Custom plotting of projected fields and power spectra 
    
    Parameters
    ----------
    plot_name : dict
        Section dictionary key from config file containing settings for plotting one figure.
    config : dict
        Entire config file.
    pfp_normal : ProjectedFieldsPlotter
        Plotter object containing all fields of normal simulations.
    pfp_onlyPF : ProjectedFieldsPlotter or None
        Plotter object containing fields of "only projected mode" simulation.
    pfp_zeroPF : ProjectedFieldsPlotter or None
        Plotter object containing fields of "zero projected mode" simulation.
    save : bool
        If true, saves all plots.
    '''    
    settings = config[plot_name]   
    keys = settings.keys()
    default = config['DEFAULT']
    io = config['IO']
    default_keys = default.keys()
    # Set default keys
    for key in keys:
        if key=='' and key in default_keys: 
            settings[key] = default[key]
    nrows, ncols = literal_eval(settings['gridsize'])
    all_snapshots = literal_eval(settings['all_snapshots'])
    snapshots = literal_eval(io['snapshots'])
    if all_snapshots:
        if type(all_snapshots) is int and all_snapshots==1:
            all_snapshots = snapshots
            if type(all_snapshots) is int: all_snapshots = list(range(all_snapshots))
    else: all_snapshots = [-1]
    for ss in all_snapshots:
        if ss > -1: print(f'Plotting snapshot {ss:03}')
        # Setup figure and ax    
        fig = plt.figure(figsize=(ncols*4, nrows*4))
        axs = []

        all_plot_params = []
        for key in keys:
            if key in default_keys: continue
            print(f'({key})')
            plot_params = settings[key].split('|')
            plot_params = [p.strip() for p in plot_params]
            if 'legend' in key:
                ax = axs[literal_eval(plot_params[0])][0]
                ax.legend()
                continue
            elif 'axeslabels' in key:
                ax_inds = literal_eval(plot_params[0])
                if type(ax_inds) is int: ax_inds = [ax_inds]
                plot_labels = literal_eval(plot_params[1])
                for i, ax_ind in enumerate(ax_inds):
                    is_ps = 'PPS' in all_plot_params[ax_ind]
                    ax = axs[ax_ind]
                    plot_label = plot_labels[i]
                    for j, p in enumerate(plot_label):
                        if is_ps: 
                            xlabel = r'$k\ \rm [h\ Mpc^{-1}]$'
                            ylabel = r'$P(k)\ \rm [h^{-3}\ Mpc^3]$'
                        else:
                            xlabel = r'$\rm h^{-1} Mpc$'
                            ylabel = r'$\rm h^{-1} Mpc$'
                        if p in ['xy', 'x']:
                            ax[j].set_xlabel(xlabel)
                            ax[j].tick_params(axis='x', reset=True)
                        if p in ['xy', 'y']:
                            ax[j].set_ylabel(ylabel)
                            ax[j].tick_params(axis='y', reset=True)
                continue
            elif 'colorbar' in key:
                ax = np.array([axs[i] for i in literal_eval(plot_params[0])])
                try: clim = literal_eval(plot_params[1])
                except: clim = None
                try: scale = literal_eval(plot_params[2])
                except: scale = 'high'
                scale_colorbar((fig, ax), clim, scale)
                continue
            elif 'projection_title' in key:
                ax = axs[literal_eval(plot_params[0])]
                pfp_normal.set_projection_labels((fig, ax))
                continue

            all_plot_params.append(plot_params)
            nrows, ncols, index = [literal_eval(p) for p in plot_params[:3]]
            if type(index) is int: index = (index, index)
            sim_type, plot_func, plot_type, plot_stat = plot_params[3:7]
            plot_func = plot_func.split()
            plot_sims = literal_eval(plot_func[1])
            plot_snapshots = literal_eval(plot_func[2])
            if ss>-1: plot_snapshots = [snapshots.index(ss)]
            try: projection = plot_func[3]
            except: projection = 'z'
            plot_func = plot_func[0]
            if len(plot_params) > 7: kwargs = literal_eval(plot_params[7])
            else: kwargs={}
            if plot_stat=='none': plot_stat = None

            if sim_type=='normal': pfp = pfp_normal
            elif sim_type=='onlyPF': pfp = pfp_onlyPF
            elif sim_type=='zeroPF': pfp = pfp_zeroPF
            else: raise ValueError("`sim_type` must be either 'normal', 'onlyPF', or 'zeroPF'.")
            
            if type(plot_sims) is int: plot_sims = list(range(plot_sims))
            if type(plot_snapshots) is int: plot_snapshots = list(range(plot_snapshots))
            if plot_sims is None: plot_sims = list(range(len(pfp.nsims)))
            if plot_snapshots is None: plot_snapshots = list(range(len(pfp.snapshots)))

            # if plot_func == 'projection':
            ax = np.array([fig.add_subplot(nrows, ncols, i) for i in range(index[0], index[1]+1)], dtype=plt.Axes)
            # else: ax = fig.add_subplot(nrows, ncols, index)
            for a in ax:
                a.tick_params(axis='both', labelbottom=False, labelleft=False)
            for ss_i in plot_snapshots:
                for s_i in plot_sims:
                    if 'label' not in kwargs and s_i==0: kwargs['label'] = key
                    elif s_i > 0: kwargs.pop('label', None)
                    if key[:2]=='__': kwargs.pop('label', None)
                    if plot_func=='plot':
                        pfp.plot(
                            sim_idx=(s_i,ss_i),
                            figax=(fig,ax),
                            plot_type=plot_type,
                            statistic=plot_stat,
                            projection=projection,
                            **kwargs
                        )
                    elif plot_func=='projections':
                        pfp.plot_projections(
                            sim_idx=(s_i,ss_i),
                            figax=(fig,ax),
                            plot_type=plot_type,
                            statistic=plot_stat,
                            **kwargs
                        )
                    elif plot_func=='all':
                        pfp.plot_all(
                            sim_idx=(s_i,ss_i),
                            figax=(fig,ax),
                            plot_type=plot_type,
                            statistic=plot_stat,
                            **kwargs
                        )
            axs.append(ax)
        if ss > -1:
            fig.suptitle(f'z = {pfp.redshifts[snapshots.index(ss)]:2.1f}')
        if save: 
            plot_out = pfp_normal.plot_dir+'/'+pfp_normal.output_dir
            plot_out += f'_{plot_name.lower().replace(" ", "_")}'
            if ss > -1: plot_out += f'_{ss:03}'
            plot_out += '.png'
            print(f'Saving plot to: {plot_out}')
            fig.savefig(plot_out)

    return fig, axs        

def scale_colorbar(figax, clim=None, scale='high'):
    fig, ax = figax

    # Find color limits of images
    if clim is None and scale in ['high', 'low']:
        for curax in ax.flatten():
            if not len(curax.get_images()) > 0:
                raise ValueError('Axes must contain images from plt.imshow')
            if len(curax.get_images()) > 0:
                cur_im = curax.get_images()[0]
            else: continue
            cur_clim = np.array(cur_im.get_clim())
            if clim is None: clim = cur_clim
            if scale=='high':
                if cur_clim[0] < clim[0]: clim[0] = cur_clim[0]
                if cur_clim[1] > clim[1]: clim[1] = cur_clim[1]
            elif scale=='low':
                if cur_clim[0] > clim[0]: clim[0] = cur_clim[0]
                if cur_clim[1] < clim[1]: clim[1] = cur_clim[1]
            else: raise ValueError("`scale` must be 'high' or 'low'.")
        clim = tuple(clim)
    elif type(scale) is int:
        curax = ax.flatten()[scale]
        try: assert len(curax.get_images()) > 0
        except: raise ValueError('Axes must contain images from plt.imshow')
        cur_im = curax.get_images()[0]
        clim = np.array(cur_im.get_clim())

    print('Setting color limits to ', clim)

    # Set color limits of images
    for curax in ax.flatten():
        if len(curax.get_images()) > 0:
            curax.get_images()[0].set_clim(clim[0], clim[1])

    # Make colorbar
    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.04, 0.7])
    # cbar = fig.colorbar(ax.flatten()[0].get_images()[0], cax=cbar_ax)
    # cbar.set_label(r'Detection Significance ($\sigma$)')
    return figax

def compute_power_spectrum(data1, Nmesh, BoxSize, data2=None, projected=None, shotcorrection=False):
    n = len(data1.shape)
    if data2 is not None:
        try: assert data2.shape == data1.shape
        except: raise ValueError('Both data arrays must have same shape')
    L = BoxSize
    N = Nmesh
    dk = 2*np.pi / L
    kmax = np.pi / L * N
    kbins = np.arange(0, kmax+dk/2, dk)
    ks = fft.fftfreq(N, 1 / (2*kmax))
    if n == 2:
        kgrid = np.meshgrid(ks, ks)
        knorm = np.sqrt(kgrid[0]**2 + kgrid[1]**2)
    elif n == 3:
        kgrid = np.meshgrid(ks, ks, ks)
        knorm = np.sqrt(kgrid[0]**2 + kgrid[1]**2 + kgrid[2]**2)
        if projected is not None:
            knorm = knorm[:,:,0]
    else: raise ValueError("Can only compute 2D or 3D power spectrum.")
    
    fdata1 = fft.fftn(data1) / N**3
    if data2 is not None:
        fdata2 = fft.fftn(data2) / N**3
    else: fdata2 = fdata1
    fdata_amp = (fdata1 * fdata2.conj()).real
    knorm = knorm.flatten()
    if n == 3 and projected is not None:
        fdata_amp = fdata_amp[:,:,0]
        n = 2
    fdata_amp = fdata_amp.flatten()

    Pkbins, _, _ = stats.binned_statistic(knorm, fdata_amp, bins=kbins)
    kvals = 0.5*(kbins[1:] + kbins[:-1])
    Pkbins *= L**n
    Pkbins[0] = 0
    kvals[0] = 0
    return kvals, Pkbins

if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read(sys.argv[1])  # e.g. simulations/config.ini
    plot_show = 0
    if len(sys.argv) > 2:
        plot_show = int(sys.argv[2]) # 1 for plt.show()

    io = config['IO']
    Nmesh = literal_eval(io['Nmesh'])
    save = literal_eval(io['save'])
    load = literal_eval(io['load'])
    prefix = io['prefix']
    sims_list = io['sims'].split('|')
    snapshots_list = io['snapshots'].split('|')
    do_ics_list = io['do_ics'].split('|')
    inputs = []
    for sims, snapshots, do_ics in zip(sims_list, snapshots_list, do_ics_list):
        inputs.append(dict(
            sims=literal_eval(sims.strip()), 
            snapshots=literal_eval(snapshots.strip()), 
            do_ics=literal_eval(do_ics.strip())
        ))

    print('Loading all fields...')
    pfp = ProjectedFieldsPlotter(
        gadget_dir=io['gadget_dir'],
        output_dir=io['output_dir'],
        plot_dir=io['plot_dir'],
        inputs=inputs,
        Nmesh=Nmesh,
        load_prefix=prefix if load else ''
    )

    if not os.path.exists(io['plot_dir']):
        os.makedirs(io['plot_dir'])

    onlypf_dir = io['onlypf_dir']
    zeropf_dir = io['zeropf_dir']
    pfp_only = None
    pfp_zero = None
    inputs = dict(
        sims=0,
        snapshots=literal_eval(snapshots_list[0].strip()),
        do_ics=literal_eval(do_ics_list[0].strip())            
    )

    if onlypf_dir:
        print(f'Loading OPM fields...')
        pfp_only = ProjectedFieldsPlotter(
            gadget_dir=io['gadget_dir'],
            output_dir=io['onlypf_dir'],
            plot_dir=io['plot_dir'],
            inputs=inputs,
            Nmesh=Nmesh,
            load_prefix=prefix if load else ''
        )
        pfp_only.mean_fields = pfp.mean_fields
        pfp.compute_statistic_field()
        pfp.compute_power_spectra()
        print()

    if zeropf_dir:
        print(f'Loading NoPM fields...')
        pfp_zero = ProjectedFieldsPlotter(
            gadget_dir=io['gadget_dir'],
            output_dir=io['zeropf_dir'],
            plot_dir=io['plot_dir'],
            inputs=inputs,
            Nmesh=Nmesh,
            load_prefix=prefix if load else ''
        )
        print()
    
    if save and not load:
        pfp.save_fields(prefix)
        try: 
            pfp_only.save_fields(prefix)
            pfp_zero.save_fields(prefix)
        except: pass
        if save == 1: exit()
    
    for plot_name in config.keys():
        if plot_name in ['IO', 'DEFAULT']: continue
        fig, ax = plot_custom(plot_name, config['DEFAULT'], pfp, pfp_only, pfp_zero)
        
        if plot_show: plt.show()
