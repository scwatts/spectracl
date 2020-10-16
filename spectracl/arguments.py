import argparse
import pathlib
import shutil


from . import __version__
from . import __program_name__


class WideHelpFormatter(argparse.HelpFormatter):

    def __init__(self, *args, **kwargs):
        terminal_width = shutil.get_terminal_size().columns
        help_width = min(terminal_width, 140)
        super().__init__(*args, **kwargs, max_help_position=help_width, width=help_width)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=WideHelpFormatter, add_help=False)
    parser.add_argument('--spectra_dir', required=True, type=pathlib.Path,
        help='Spectra directory')
    parser.add_argument('-v', '--version', action='version', version=f'{__program_name__} {__version__}',
        help='Show version number and exit')
    parser.add_argument('-h', '--help', action='help',
        help='Show this help message and exit')
    args = parser.parse_args()
    if not args.spectra_dir.exists():
        parser.error(f'Spectra directory {args.spectra_dir} does not exist')
    return args
