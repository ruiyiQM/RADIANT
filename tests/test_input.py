"""Exercise the actual TOML single-state entry point in an isolated directory."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from radiant.input import run_toml


INPUT = '''[system]
atoms = "H 0 0 0; H 0 0 1.4"
unit = "Bohr"
masses = [1836.0, 1836.0]
momenta = [[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]]

[backend]
model = "rhf"
basis = "sto-3g"

[state]
initial = "scf"
alpha = 2.0

[propagation]
variant = "cn_accelerated"
tolerance = 1e-10
max_iterations = 80

[trajectory]
dt = 0.1
steps = 2
n_substeps = 2

[output]
file = "trajectory.npz"
'''


class InputTests(unittest.TestCase):
    def test_toml_runs_and_writes_only_in_its_output_directory(self):
        for variant in ("cn_accelerated", "pt_cn", "pt_im"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                text=INPUT.replace("cn_accelerated", variant)
                if variant.startswith("pt"):
                    text=text.replace("max_iterations = 80","max_iterations = 80\nhistory = 8")
                path=root/"input.toml"; path.write_text(text)
                run=run_toml(path)
                self.assertEqual(run.time,.2)
                self.assertEqual(len(run.frames),2)
                with np.load(root/"trajectory.npz",allow_pickle=False) as data:
                    self.assertTrue(data["completed"])
                    assert_allclose(data["time"],[.1,.2],atol=0,rtol=0)
                self.assertEqual({p.name for p in root.iterdir()},{"input.toml","trajectory.npz"})

    def test_units_unknown_fields_and_unprepared_excited_state_are_rejected(self):
        for text in (INPUT.replace('unit = "Bohr"','unit = "angstrom"'),
                     INPUT.replace('initial = "scf"','initial = "excited"'),
                     INPUT.replace('alpha = 2.0','alpha = 2.0\nmu = 3.0')):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"input.toml";path.write_text(text)
                with self.assertRaises(ValueError):
                    run_toml(path)
                self.assertFalse((path.parent/"trajectory.npz").exists())


if __name__ == "__main__":
    unittest.main()
