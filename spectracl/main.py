import pathlib
import pickle
import sys
from typing import List

import pandas as pd
import numpy as np

from . import arguments
from . import spectra


def entry():
    # Get command line arguments
    args = arguments.get_args()

    # Read in sample data
    sample_data = dict()
    if args.sample_sheet_fp:
        sample_data = read_sample_sheet(args.sample_sheet_fp)

    # Load model and selected features
    with args.model_fp.open('rb') as fh:
        model = pickle.load(fh)
    with args.features_fp.open('r') as fh:
        features = [int(line.rstrip()) for line in fh]
    # Discover fid files
    fid_fps = list(args.spectra_dir.rglob('fid'))
    if len(fid_fps) == 0:
        print(f'error: did not find any fid files in {args.spectra_dir}', file=sys.stderr)
        sys.exit(1)

    for result in compute(features, fid_fps, model, sample_data):
        print(*result, sep='\t')


def compute(
        features: List[int],
        fid_fps: List[pathlib.Path],
        model,
        sample_data: dict,
) -> List:
    # Read in spectrum data and sort according to sample sheet, if provided
    spectra_grouped = dict()
    sp_identifiers = ['uid', 'sample_name', 'full_name', 'semi_unique_name']
    for sp in (spectra.Spectra(fid_fp) for fid_fp in fid_fps):
        for sp_identifier in sp_identifiers:
            if identifier := sample_data.get(getattr(sp, sp_identifier)):
                if identifier not in spectra_grouped:
                    spectra_grouped[identifier] = list()
                spectra_grouped[identifier].append(sp)
                break
        else:
            assert sp.unique_name not in spectra_grouped
            spectra_grouped[sp.unique_name] = [sp]

    # Process spectra
    sp_data = pd.DataFrame(
        np.zeros((len(spectra_grouped), spectra.MASS_BINS.size-1)),
        columns=spectra.MASS_BINS[:-1]
    )
    for i, (sample_name, sps) in enumerate(spectra_grouped.items()):
        print(f'info: processing sample \'{sample_name}\' ({i+1} of {len(spectra_grouped)})', file=sys.stderr)
        if len(sps) > 1:
            sp_data_sample_list = [spectra.process(sp) for sp in sps]
            sp_data_sample = pd.DataFrame(sp_data_sample_list)
            sp_data.iloc[i] = spectra.average_spectra(sp_data_sample)
        else:
            sp_data.iloc[i] = spectra.process(sps[0])

    # Select specific features then classify
    sp_data = sp_data[features]
    pred_probs = model.predict_proba(sp_data)
    for sample_name, probs in zip(spectra_grouped.keys(), pred_probs):
        species_index = np.argmax(probs)
        species_prob = round(probs[species_index] * 100, 2)
        species = model.classes_[species_index]
        yield sample_name, species, species_prob


def read_sample_sheet(fp):
    data = dict()
    required_columns = ('spectrum_id', 'sample_id')
    with fp.open('r') as fh:
        line_token_gen = (line.rstrip().split('\t') for line in fh)
        header_tokens = next(line_token_gen)
        assert all(rc in header_tokens for rc in required_columns)
        for line_tokens in line_token_gen:
            entry = {k: v for k, v in zip(header_tokens, line_tokens)}
            data[entry['spectrum_id']] = entry['sample_id']
    return data
