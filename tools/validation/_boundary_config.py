"""Load the upstream boundary-control cases without changing their source files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def configuration(case, output=None):
    path = ROOT / 'differentiability-data/input' / f'{case}.json'
    config = json.loads(path.read_text())
    config['root_path'] = str(path)
    for geometry in config['geometry']:
        geometry['mesh'] = str((path.parent / geometry['mesh']).resolve())
    config['solver']['max_threads'] = 1
    config['solver']['linear']['solver'] = 'Eigen::SimplicialLDLT'
    config['output'] = {
        'directory': '' if output is None else str(output), 'json': '',
        'log': {'level': 'warning'}, 'paraview': {'file_name': ''},
        'advanced': {'save_time_sequence': False},
    }
    return config
