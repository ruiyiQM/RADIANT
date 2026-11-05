"""Moving-basis endpoint and whole-step rollback regressions."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from radiant.backends.pyscf import PySCFProvider
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions
from radiant.core.state import NuclearState
from radiant.driver import Trajectory, follow_path
from test_pyscf import mean_field
from toy_provider import ToyProvider


class ExecutionTests(unittest.TestCase):
    def test_moving_pyscf_endpoint_and_reusable_snapshots(self):
        provider = PySCFProvider(mean_field("rhf"))
        run = LocalExecution([provider], alphas=[2.])
        before = run.snapshot()
        end = run.positions.copy(); end[1,2] += .01
        first = run.advance(end,.1,n_substeps=2)
        final_density = run.states[0].density()
        final_force = first.observation.forces.copy()
        for _ in range(2):
            run.restore(before)
            self.assertEqual(run.time,0.)
            again=run.advance(end,.1,n_substeps=2)
            assert_allclose(run.states[0].density(),final_density,atol=1e-12,rtol=0)
            assert_allclose(again.observation.forces,final_force,atol=1e-12,rtol=0)
            self.assertLess(run.states[0].orthogonality_error(run.providers[0].overlap),1e-12)
        assert_allclose(run.positions,end,atol=0)
        self.assertEqual(first.observation.time,.1)

    def test_second_state_and_late_substep_failure_restore_everything(self):
        for failing_branch, threshold in ((1,.1),(0,.3)):
            run=LocalExecution([ToyProvider(0),ToyProvider(1)])
            run.providers[failing_branch].failure_time=threshold
            before=run.snapshot()
            with self.assertRaisesRegex(RuntimeError,"Injected"):
                run.advance([[.2,0,0]],.4,n_substeps=4)
            self.assertEqual(run.time,0.)
            assert_allclose(run.positions,[[0,0,0]],atol=0)
            for state, saved in zip(run.states,before.states):
                assert_allclose(state.density(),saved.density(),atol=0)
            self.assertTrue(all(p.cache is None for p in run.providers))
            for a,b in zip(run.gauge,before.gauge):
                assert_allclose(a.coefficients[0],b.coefficients[0],atol=0)
            run.providers[failing_branch].failure_time=None
            reference=LocalExecution([ToyProvider(0),ToyProvider(1)])
            expected=reference.advance([[.2,0,0]],.4,n_substeps=4)
            actual=run.advance([[.2,0,0]],.4,n_substeps=4)
            assert_allclose(actual.overlap_raw,expected.overlap_raw,atol=1e-12,rtol=0)

    def test_alpha_only_scales_electronics_and_substeps_use_physical_time(self):
        ToyProvider.trace=[]
        run=LocalExecution([ToyProvider(force=.1)],alphas=[10.],time=3.)
        result=run.advance([[.2,0,0]],.4,n_substeps=4)
        self.assertEqual({round(t,12) for _,t,_,_ in ToyProvider.trace},{3.,3.1,3.2,3.3,3.4})
        for _,_,positions,_ in ToyProvider.trace:
            assert_allclose(positions,[[.1,0,0]])
        self.assertAlmostEqual(result.observation.energies[0],-.02)
        assert_allclose(result.observation.forces,[[[.1,0,0]]])
        self.assertAlmostEqual(result.observation.time,3.4)

    def test_nuclear_verlet_and_failure_output_keep_only_accepted_frames(self):
        execution=LocalExecution([ToyProvider(force=.1)])
        nuclei=NuclearState([[0,0,0]],[[0,0,0]],[2.])
        run=Trajectory(execution,nuclei)
        run.step(.4,n_substeps=2)
        assert_allclose(run.nuclei.positions,[[.004,0,0]],atol=1e-15)
        assert_allclose(run.nuclei.momenta,[[.04,0,0]],atol=1e-15)
        self.assertAlmostEqual(run.frames[0]["total_energy"],0.)
        saved=deepcopy(run.nuclei)
        run.execution.providers[0].failure_time=.6
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"trajectory.npz"
            with self.assertRaises(RuntimeError):
                run.run(2,.4,n_substeps=2,output=path)
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(data["completed_steps"],1)
                self.assertEqual(data["failed_step"],2)
                self.assertFalse(data["completed"])
                self.assertEqual(data["failure_reason"],"RuntimeError")
                assert_allclose(data["time"],[.4])
        assert_allclose(run.nuclei.positions,saved.positions,atol=0)
        assert_allclose(run.nuclei.momenta,saved.momenta,atol=0)
        self.assertEqual(run.time,.4)
        self.assertEqual(execution.time,.4)

    def test_fixed_path_uses_explicit_times_and_first_geometry(self):
        run=LocalExecution([ToyProvider()])
        path=np.zeros((4,1,3));path[:,0,0]=[0,.1,.4,.5]
        frames=follow_path(run,path,[0.,.2,.5,.9],n_substeps=2)
        assert_allclose([f.observation.time for f in frames],[.2,.5,.9],atol=1e-15)
        assert_allclose(run.positions,path[-1],atol=0)
        foreign=LocalExecution([ToyProvider()]).snapshot()
        with self.assertRaises(ValueError):
            run.restore(foreign)
        with self.assertRaises(ValueError):
            follow_path(run,path,[0.,.2,.5,.9])
        with self.assertRaises(ValueError):
            LocalExecution([ToyProvider()],options=CNOptions(),alphas=[0.])

    def test_integer_nuclear_step_clock(self):
        execution=LocalExecution([ToyProvider(rate=0.)])
        run=Trajectory(execution,NuclearState([[0,0,0]],[[0,0,0]],[1.]))
        run.run(100,.4)
        assert_allclose([f["time"] for f in run.frames],np.arange(1,101)*.4,atol=0,rtol=0)
        self.assertEqual(run.time,40.)
        self.assertEqual(execution.time,40.)

    def test_output_failure_rolls_back_and_preserves_previous_file(self):
        execution=LocalExecution([ToyProvider(force=.1)])
        run=Trajectory(execution,NuclearState([[0,0,0]],[[0,0,0]],[2.]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"trajectory.npz"
            run.run(1,.4,output=path)
            previous=path.read_bytes()
            with self.assertRaises(ValueError):
                run.run(1,0.,output=path)
            self.assertEqual(path.read_bytes(),previous)
            saved=deepcopy(run.nuclei)
            with patch("radiant.driver.os.replace",side_effect=OSError("Injected output failure")):
                with self.assertRaisesRegex(OSError,"output failure"):
                    run.run(1,.4,output=path)
            self.assertEqual(path.read_bytes(),previous)
            self.assertEqual(list(Path(folder).iterdir()),[path])
        self.assertEqual(run.time,.4)
        self.assertEqual(execution.time,.4)
        self.assertEqual(len(run.frames),1)
        assert_allclose(run.nuclei.positions,saved.positions,atol=0)
        assert_allclose(run.nuclei.momenta,saved.momenta,atol=0)


if __name__ == "__main__":
    unittest.main()
