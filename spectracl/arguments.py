import argparse
import pathlib
import shutil


from . import __version__
from . import __program_name__


PKG_DATA_DIR = pathlib.Path(__file__).parent / 'data'
PKG_MODEL_FP = PKG_DATA_DIR / 'model.bin'
PKG_FEATURES_FP = PKG_DATA_DIR / 'features_selected.txt'


class WideHelpFormatter(argparse.HelpFormatter):

    def __init__(self, *args, **kwargs):
        terminal_width = shutil.get_terminal_size().columns
        help_width = min(terminal_width, 140)
        super().__init__(*args, **kwargs, max_help_position=help_width, width=help_width)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=WideHelpFormatter, add_help=False)
    parser.add_argument(
        '--spectra_dir',
        required=True,
        type=pathlib.Path,
        help='Spectra directory'
    )
    parser.add_argument(
        '--sample_sheet_fp',
        required=False,
        type=pathlib.Path,
        help='Sample sheet containing information to group and average spectrum'
    )
    parser.add_argument(
        '--model_fp',
        required=False,
        type=pathlib.Path,
        default=PKG_MODEL_FP,
        help='Model file used for classification'
    )
    parser.add_argument(
        '--features_fp',
        required=False,
        type=pathlib.Path,
        default=PKG_FEATURES_FP,
        help='File containing features to select for classification'
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'{__program_name__} {__version__}',
        help='Show version number and exit'
    )
    parser.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit'
    )
    args = parser.parse_args()
    if not args.spectra_dir.exists():
        parser.error(f'Spectra directory {args.spectra_dir} does not exist')
    if args.sample_sheet_fp and not args.sample_sheet_fp.exists():
        parser.error(f'Sample sheet file {args.sample_sheet} does not exist')
    if not args.model_fp.exists():
        if args.model_fp == PKG_MODEL_FP:
            msg_p1 = f'Missing the model file provided with package ({args.model_fp}).'
            msg_p2 = 'Please set path to the model file using --model_fp'
            parser.error(f'{msg_p1} {msg_p2}')
        else:
            parser.error(f'Input model file {args.model_fp} does not exist')
    if not args.features_fp.exists():
        if args.features_fp == PKG_FEATURES_FP:
            msg_p1 = f'Missing the features file provided with package ({args.features_fp}).'
            msg_p2 = 'Please set path to the features file using --features_fp'
            parser.error(f'{msg_p1} {msg_p2}')
        else:
            parser.error(f'Input features file {args.features_fp} does not exist')
    return args
