import pandas as pd
import numpy as np
import scipy.signal
import scipy.stats


MASS_BINS = np.arange(1950, 21000, 1)


class Spectra:

    def __init__(self, fid_fp):
        # Set input filepaths
        self.fid_fp = fid_fp
        self.acqu_fp = self.fid_fp.parent / 'acqu'
        if not self.acqu_fp.exists():
            print(f'error: fid file {self.fid_fp} is missing accompanying acqu file', file=sys.stderr)
            sys.exit(1)
        # Set basic data
        self.acqu_data = read_acqu_data(self.acqu_fp)
        dtype = '<i4' if self.acqu_data['byte_order'] == 'little' else '>i4'
        self.intensities = np.fromfile(self.fid_fp, dtype=dtype)
        self.tofs = np.arange(self.acqu_data['time_periods']) * self.acqu_data['time_delta'] + self.acqu_data['time_delay']
        self.masses = get_masses(self.tofs, self.acqu_data)
        # Misc
        self.uid = self.acqu_data['spectra_uid']
        self.position = self.acqu_data['position'].strip('<>')
        self.sample_number = self.fid_fp.parts[-4]
        self.sample_name = self.fid_fp.parts[-6]
        self.full_name = f'{self.sample_name}.{self.position}'
        self.semi_unique_name = f'{self.full_name}.{self.sample_number}'
        self.unique_name = f'{self.full_name}.{self.sample_number}.{self.uid}'


def process(sp):
    # Apply square root transformation and smooth intensities
    sp.intensities = np.sqrt(sp.intensities)
    sp.intensities = scipy.signal.savgol_filter(sp.intensities, 21, 3)

    # Remove baseline and calibrate intensities
    baseline = get_baseline(sp.intensities)
    sp.intensities = sp.intensities - baseline
    sp.intensities = calibrate_intensities(sp.intensities, sp.masses)

    # Remove nans and replace negative intensities with zero
    nan_indices = np.argwhere(np.isnan(sp.intensities))
    if nan_indices.size == sp.intensities.size:
        sp.intensities = np.zeros(sp.intensities.size)
    elif nan_indices.size > 0:
        sp.intensities = np.delete(sp.intensities, nan_indices)
        sp.masses = np.delete(sp.masses, nan_indices)
    sp.intensities[sp.intensities < 0] = 0

    # Bin intensities and return bin means
    intensities_binned = scipy.stats.binned_statistic(
        sp.masses,
        sp.intensities,
        statistic='mean',
        bins=MASS_BINS
    )
    np.nan_to_num(intensities_binned.statistic, copy=False, nan=0)
    return intensities_binned.statistic


def read_acqu_data(acqu_fp):
    acqu_fields = {
        'ID_raw': {'name': 'spectra_uid', 'vtype': str},
        'PATCHNO': {'name': 'position', 'vtype': str},
        'BYTORDA': {'name': 'byte_order', 'vtype': int},
        'DELAY': {'name': 'time_delay', 'vtype': int},
        'DW':  {'name': 'time_delta', 'vtype': int},
        'TD':  {'name': 'time_periods', 'vtype': int},
        'ML1': {'name': 'calibration_const_1', 'vtype': float},
        'ML2': {'name': 'calibration_const_2', 'vtype': float},
        'ML3': {'name': 'calibration_const_3', 'vtype': float},
    }
    acqu_data = dict()
    with acqu_fp.open('r') as fh:
        for line in fh:
            if not line.startswith('##$'):
                continue
            var, val = line.lstrip('##$').rstrip().split('= ')
            if var in acqu_fields:
                name = acqu_fields[var]['name']
                vtype = acqu_fields[var]['vtype']
                assert name not in acqu_data
                acqu_data[name] = vtype(val)
    for d in acqu_fields.values():
        assert d['name'] in acqu_data, f"Missing {d['name']} in {acqu_fp}" 
    # Apply further processing to spectra_uid and byte_order
    acqu_data['spectra_uid'] = acqu_data['spectra_uid'].strip('<>')
    acqu_data['byte_order'] = 'little' if acqu_data['byte_order'] == 0 else 'big'
    return acqu_data


def get_masses(tofs, acqu_data):
    # https://dx.doi.org/10.1186%2F1471-2105-7-403
    a = acqu_data['calibration_const_3']
    b = np.sqrt(1e12 / acqu_data['calibration_const_1'])
    c = np.array(acqu_data['calibration_const_2'] - tofs)
    if a == 0:
        masses = (c * c) / (b * b)
    else:
        masses = ((-b + np.sqrt( (b * b) - (4 * a * c)) ) / ( 2 * a )) ** 2
    return masses


def get_baseline(intensities, *, niter=100):
    # https://doi.org/10.1016/0168-583X(88)90063-8
    # https://doi.org/10.1016/j.nima.2008.11.132
    y = intensities.copy()
    n = intensities.size
    z = np.zeros(n)
    # Inner loop vectorised by rolling arrays to align pairs
    for p in reversed(range(1, niter+1)):
        a1 = y[p:n-p]
        a2 = (np.roll(y, p)[p:n-p] + np.roll(y, -p)[p:n-p]) / 2
        y[p:n-p] = np.minimum(a1, a2)
    return y


def calibrate_intensities(intensities, masses):
    scale = sum((intensities[:-1] + intensities[1:]) / 2 * np.diff(masses))
    return intensities / scale


def average_spectra(sp_data):
    sp_data_interp = pd.DataFrame().reindex_like(sp_data)
    for i in range(sp_data.shape[0]):
        f = scipy.interpolate.interp1d(MASS_BINS[:-1], sp_data.iloc[i])
        sp_data_interp.iloc[i] = f(MASS_BINS[:-1])
    return pd.DataFrame([np.mean(sp_data_interp, axis=0)])
