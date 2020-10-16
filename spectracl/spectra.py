import numpy as np
import scipy.signal
import scipy.stats


def process(fid_fp, acqu_fp, mass_bins):
    # Read in fid data and acqu data
    acqu_data = read_acqu_data(acqu_fp)
    dtype = '<i4' if acqu_data['byte_order'] == 'little' else '>i4'
    intensities = np.fromfile(fid_fp, dtype=dtype)

    # Get masses
    tofs = np.arange(acqu_data['time_periods']) * acqu_data['time_delta'] + acqu_data['time_delay']
    masses = get_masses(tofs, acqu_data)

    # Apply square root transformation and smooth intensities
    intensities = np.sqrt(intensities)
    intensities = scipy.signal.savgol_filter(intensities, 21, 3)

    # Remove baseline and calibrate intensities
    baseline = get_baseline(intensities)
    intensities = intensities - baseline
    intensities = calibrate_intensities(intensities, masses)

    # Remove nans and replace negative intensities with zero
    nan_indices = np.argwhere(np.isnan(intensities))
    if nan_indices.size == intensities.size:
        intensities = np.zeros(intensities.size)
    elif nan_indices.size > 0:
        intensities = np.delete(intensities, nan_indices)
        masses = np.delete(masses, nan_indices)
    intensities[intensities < 0] = 0

    # Bin intensities and return bin means
    intensities_binned = scipy.stats.binned_statistic(masses, intensities, statistic='mean', bins=mass_bins)
    np.nan_to_num(intensities_binned.statistic, copy=False, nan=0)
    return acqu_data['spectra_uid'], intensities_binned.statistic


def read_acqu_data(acqu_fp):
    acqu_fields = {
        'ID_raw': {'name': 'spectra_uid', 'vtype': str},
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
    assert all(d['name'] in acqu_data for d in acqu_fields.values())
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
