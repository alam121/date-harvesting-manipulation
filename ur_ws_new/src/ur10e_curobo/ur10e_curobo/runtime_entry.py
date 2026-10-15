"""Launch the motion node with its isolated cuRobo v2 environment."""
import os
from pathlib import Path
import sys


def main():
    python = Path(os.environ.get(
        'UR10E_CUROBO_V2_PYTHON', '~/.venvs/datepalm-curobo-v2/bin/python')).expanduser()
    if not python.is_file():
        raise RuntimeError(
            'cuRobo v2 environment is missing. Run bin/setup_curobo_v2 first, '
            'or set UR10E_CUROBO_V2_PYTHON to its Python executable.')
    os.execv(str(python), [str(python), '-m', 'ur10e_curobo.main', *sys.argv[1:]])
