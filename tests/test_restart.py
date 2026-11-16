"""Numeric restart equivalence, malformed inputs and atomic-write regressions."""

from copy import deepcopy
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from pyscf import scf

from radiant.backends.pyscf import PySCFProvider
from radiant.backends.gpu_pyscf import GPU4PySCFProvider
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions
from radiant.core.state import NuclearState, OrbitalState
from radiant.driver import Trajectory
from radiant.methods.mash import MASH
from radiant.restart import load_checkpoint, save_checkpoint
from test_gpu_provider import mean_field


def make_run(mf, variant="pt_im", pair=True, backend=PySCFProvider):
    provider = backend(mf)
    states = [provider.initial_state()]
    if pair:
        excited = (OrbitalState((mf.mo_coeff[0][:, :2], mf.mo_coeff[1][:, 1:2]), states[0].occupations)
                   if provider.unrestricted else OrbitalState((mf.mo_coeff[:, :2],), ([1., 1.],)))
        states.append(excited)
    providers = [deepcopy(provider) for _ in states]
    execution = LocalExecution(providers, states, options=CNOptions(variant, 1e-11, 120))
    momentum = np.zeros_like(provider.positions); momentum[-1, 2] = .5
    nuclei = NuclearState(provider.positions, momentum, [7294., 1836.])
    return Trajectory(execution, nuclei, method=MASH([.6, 0., .8]) if pair else None, hop_tolerance=1e-4 if pair else None)

class RestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = (mean_field(False), mean_field(True))

    def compare(self, left, right):
        self.assertEqual(left.time, right.time)
        self.assertEqual(left.initial_time, right.initial_time)
        self.assertEqual(left.nuclear_dt, right.nuclear_dt)
        self.assertEqual(len(left.frames), len(right.frames))
        assert_allclose(left.nuclei.positions, right.nuclei.positions, atol=2e-12, rtol=0)
        assert_allclose(left.nuclei.momenta, right.nuclei.momenta, atol=2e-12, rtol=0)
        for a, b in zip(left.execution.states, right.execution.states):
            assert_allclose(a.density(), b.density(), atol=2e-12, rtol=0)
        if left.method is not None:
            assert_allclose(left.method.coefficients, right.method.coefficients, atol=2e-12, rtol=0)
            for a, b in zip(left.execution.gauge, right.execution.gauge):
                for ca, cb in zip(a.coefficients, b.coefficients):
                    assert_allclose(ca, cb, atol=2e-12, rtol=0)
        for a, b in zip(left.frames, right.frames):
            for key in a:
                assert_allclose(a[key], b[key], atol=2e-12, rtol=0)
        self.assertEqual(left.hop_events, right.hop_events)

    def test_resumed_rks_uks_and_single_state_match_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"restart.npz"
            for mf, pair in ((self.fields[0], False), (self.fields[0], True), (self.fields[1], True)):
                for variant in ("cn_accelerated", "pt_cn", "pt_im"):
                    with self.subTest(unrestricted=isinstance(mf, scf.uhf.UHF), pair=pair, variant=variant):
                        full, split = make_run(mf, variant, pair), make_run(mf, variant, pair)
                        full.run(4, .02)
                        split.run(2, .02, checkpoint=path)
                        templates = make_run(mf, variant, pair).execution.providers
                        with patch.object(type(mf), "kernel", side_effect=AssertionError("SCF during restart")):
                            resumed = load_checkpoint(path, templates)
                        self.compare(split, resumed)
                        resumed.run(2, .02, checkpoint=path)
                        self.compare(full, resumed)

    def test_empty_and_failed_step_checkpoints_keep_accepted_state(self):
        mf = self.fields[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"restart.npz"
            run = make_run(mf)
            save_checkpoint(run, path)
            self.compare(run, load_checkpoint(path, run.execution.providers))
            run.run(1, .02)
            saved = deepcopy(run.nuclei)
            with patch.object(PySCFProvider, "fock", side_effect=RuntimeError("Injected backend failure")):
                with self.assertRaisesRegex(RuntimeError, "Injected"):
                    run.run(1, .02, checkpoint=path)
            loaded = load_checkpoint(path, make_run(mf).execution.providers)
            self.assertEqual(len(loaded.frames), 1)
            self.assertEqual(loaded.failure[0], 2)
            assert_allclose(loaded.nuclei.positions, saved.positions, atol=0, rtol=0)
            loaded.run(1, .02)
            full = make_run(mf); full.run(2, .02)
            self.compare(full, loaded)

    def test_invalid_data_and_changed_model_are_rejected_without_mutating_templates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"restart.npz"
            run = make_run(self.fields[0]); run.run(1, .02)
            save_checkpoint(run, path)
            with np.load(path, allow_pickle=False) as source:
                original = {k: source[k] for k in source.files}
            templates = make_run(self.fields[0]).execution.providers
            positions = [p.positions.copy() for p in templates]
            for key in ("state0_c0", "gauge1_c0", "positions", "forces", "frame_time"):
                values = {k: v.copy() for k, v in original.items()}
                values[key].flat[0] = np.nan
                np.savez(path, **values)
                with self.subTest(field=key), self.assertRaises(ValueError):
                    load_checkpoint(path, templates)
            for key in ("frame_energies", "frame_active_state", "frame_spin", "frame_kinetic_energy", "frame_total_energy"):
                values = {k: v.copy() for k, v in original.items()}
                values[key].flat[0] += 1
                np.savez(path, **values)
                with self.subTest(inconsistent_history=key), self.assertRaisesRegex(ValueError, 'accepted state'):
                    load_checkpoint(path, templates)
            values = original.copy(); info = json.loads(str(values['metadata'].item())); info['format'] = 2
            values['metadata'] = np.array(json.dumps(info)); np.savez(path, **values)
            with self.assertRaises(ValueError):
                load_checkpoint(path, templates)
            np.savez(path, **original)
            changed = deepcopy(templates); changed[0].mf.xc = 'lda,vwn'
            with self.assertRaisesRegex(ValueError, 'settings'):
                load_checkpoint(path, changed)
            for provider, saved in zip(templates, positions):
                assert_allclose(provider.positions, saved, atol=0, rtol=0)

    def test_failed_atomic_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"restart.npz"
            run = make_run(self.fields[0]); save_checkpoint(run, path)
            original = path.read_bytes()
            run.run(1, .02)
            with patch('radiant.restart.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    save_checkpoint(run, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            self.assertEqual(load_checkpoint(path, run.execution.providers).time, 0.)

    def test_hop_history_keeps_native_scalar_types_after_restart(self):
        run = make_run(self.fields[0]); run.run(1, .02)
        run.hop_events.append((1, .01, 0, 1, 'accepted', .2, .03))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            save_checkpoint(run, path)
            loaded = load_checkpoint(path, run.execution.providers)
            self.assertEqual(loaded.hop_events, run.hop_events)
            self.assertIs(type(loaded.hop_events[0][0]), int)
            self.assertEqual(json.loads(json.dumps(loaded.hop_events))[0][4], 'accepted')


@unittest.skipUnless(find_spec("cupy") and find_spec("gpu4pyscf"), "Requires a real GPU environment")
class GPUCheckpointTests(unittest.TestCase):
    def test_gpu_pair_restart_matches_uninterrupted_without_serializing_device_objects(self):
        for mf in (mean_field(False), mean_field(True)):
            with self.subTest(kind=type(mf).__name__), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"restart.npz"
                full = make_run(mf, backend=GPU4PySCFProvider)
                split = make_run(mf, backend=GPU4PySCFProvider)
                full.run(4, .02)
                split.run(2, .02, checkpoint=path)
                with np.load(path, allow_pickle=False) as arrays:
                    self.assertFalse(any(arrays[k].dtype.hasobject for k in arrays.files))
                templates = [GPU4PySCFProvider(mf), GPU4PySCFProvider(mf)]
                resumed = load_checkpoint(path, templates)
                self.assertIsNot(resumed.execution.providers[0]._gpu_mf, split.execution.providers[0]._gpu_mf)
                resumed.run(2, .02, checkpoint=path)
                assert_allclose(resumed.nuclei.positions, full.nuclei.positions, atol=2e-12, rtol=0)
                assert_allclose(resumed.nuclei.momenta, full.nuclei.momenta, atol=2e-12, rtol=0)
                assert_allclose(resumed.method.coefficients, full.method.coefficients, atol=2e-12, rtol=0)
                for left, right in zip(resumed.execution.states, full.execution.states):
                    assert_allclose(left.density(), right.density(), atol=2e-12, rtol=0)
                self.assertEqual(len(resumed.frames), 4)
                self.assertEqual(resumed.time, full.time)


if __name__ == '__main__':
    unittest.main()
