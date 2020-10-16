import pathlib
import pickle
import sys


import pandas as pd
import numpy as np


from . import arguments
from . import spectra


def entry():
    # Get command line arguments
    args = arguments.get_args()

    # Load model and species map
    model_fp = pathlib.Path(__file__).parent / 'data/model.bin'
    species_map_fp = pathlib.Path(__file__).parent / 'data/species_map.tsv'
    with model_fp.open('rb') as fh:
        model = pickle.load(fh)
    with species_map_fp.open('r') as fh:
        line_token_gen = (line.rstrip().split('\t') for line in fh)
        species_map = {int(n): s for s, n in line_token_gen}

    # Discover fid files
    fid_fps = list(args.spectra_dir.rglob('fid'))
    if len(fid_fps) == 0:
        print(f'error: did not find any fid files in {args.spectra_dir}', file=sys.stderr)
        sys.exit(1)

    # Process spectra
    mass_bins = np.arange(1950, 21000, 10)
    sp_data = pd.DataFrame(np.zeros((len(fid_fps), mass_bins.size-1)), columns=mass_bins[:-1])
    spectra_uids = list()
    for i, fid_fp in enumerate(fid_fps):
        print(f'info: processing spectra {i+1} of {len(fid_fps)}', file=sys.stderr)
        acqu_fp = fid_fp.parent / 'acqu'
        if not acqu_fp.exists():
            print(f'error: fid file {fid_fp} is missing accompanying acqu file', file=sys.stderr)
            sys.exit(1)
        spectra_uid, sp_data.iloc[i] = spectra.process(fid_fp, acqu_fp, mass_bins)
        spectra_uids.append(spectra_uid)

    # Classify spectra
    pred_probs = model.predict_proba(sp_data)
    for spectra_uid, probs in zip(spectra_uids, pred_probs):
        species_index = np.argmax(probs)
        species_prob = round(probs[species_index] * 100, 2)
        species = species_map[species_index]
        print(spectra_uid, species, species_prob, sep='\t')


if __name__ == '__main__':
    entry()
