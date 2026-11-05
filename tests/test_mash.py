# MIT License
#
# Copyright (c) 2026 Ruiyi Zhou and Jeremy O. Richardson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Paper overlap, public spin, signed FD and hop/failure regressions."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

from radiant.core.coupling import NACResult, PaperFD, ZeroNACError
from radiant.core.execution import LocalExecution
from radiant.core.overlap import GaugeFailure, dto_pair, overlap_coupling, pair_overlap
from radiant.core.state import NuclearState, OrbitalState
from radiant.driver import Trajectory
from radiant.methods.mash import MASH, coefficients_to_spin, legacy_to_public_spin, spin_to_coefficients
from toy_provider import ToyProvider


class OverlapTests(unittest.TestCase):
    def test_frozen_paper_singlet_fixture_including_all_open_closed_cases(self):
        fixture=json.loads((Path(__file__).parent/"fixtures/paper_overlap.json").read_text())
        def decode(a):
            return np.array(a["real"])+1j*np.array(a["imag"])
        left=tuple(OrbitalState((decode(c),),(f,)) for c,f in zip(fixture["left"],fixture["occupations"]))
        right=tuple(OrbitalState((decode(c),),(f,)) for c,f in zip(fixture["right"],fixture["occupations"]))
        assert_allclose(pair_overlap(left,right,decode(fixture["cross"])),decode(fixture["expected"]),atol=1e-12,rtol=0)

    def test_dto_phases_and_central_reference_are_invariant(self):
        execution=LocalExecution([ToyProvider(0),ToyProvider(1)])
        central=deepcopy(execution.gauge)
        rng=np.random.default_rng(57)
        original=np.linalg.eigh
        def phased_eigh(matrix):
            values,vectors=original(matrix)
            return values,vectors*np.exp(1j*rng.uniform(-np.pi,np.pi,len(values)))
        for _ in range(8):
            states=tuple(OrbitalState((s.coefficients[0]*np.exp(1j*rng.uniform(-np.pi,np.pi,len(s.occupations[0]))),),s.occupations) for s in execution.states)
            with patch("radiant.core.overlap.np.linalg.eigh",side_effect=phased_eigh):
                result=dto_pair(states,np.eye(2),reference=central,cross=np.eye(2),align_all=True)
            for a,b in zip(result,central):
                assert_allclose(a.coefficients[0],b.coefficients[0],atol=1e-12,rtol=0)

    def test_time_overlap_is_distinct_from_same_time_gram(self):
        run=LocalExecution([ToyProvider(0),ToyProvider(1)])
        before=run.snapshot()
        result=run.advance([[.1,0,0]],.4)
        after=run.snapshot()
        gram=run.overlap(after,after)
        assert_allclose(gram,gram.conj().T,atol=1e-12,rtol=0)
        assert_allclose(overlap_coupling(gram,.4)[2],0,atol=1e-12,rtol=0)
        self.assertGreater(np.linalg.norm(overlap_coupling(result.overlap_raw,.4)[2]),1e-3)
        cross=ToyProvider.cross_overlap(before.providers[0].mf.mol,after.providers[0].mf.mol)
        assert_allclose(result.overlap_raw,pair_overlap(before.gauge,after.gauge,cross),atol=1e-12,rtol=0)

    def test_complex_diagonal_terms_and_gauge_covariance(self):
        raw=np.array([[.9+.1j,.1+.2j],[-.15+.2j,.8-.3j]])
        c=np.array([np.sqrt(.3),np.exp(.8j)*np.sqrt(.7)])
        first=MASH(coefficients_to_spin(c));second=MASH(coefficients_to_spin(c))
        gl=np.diag(np.exp(1j*np.array([.2,-.7])))
        gr=np.diag(np.exp(1j*np.array([-.8,.1])))
        second.coefficients=gl.conj().T@first.coefficients
        first.propagate([0,.3],[.1,.5],raw,.4)
        second.propagate([0,.3],[.1,.5],gl.conj().T@raw@gr,.4)
        assert_allclose(second.coefficients,gr.conj().T@first.coefficients,atol=1e-12,rtol=0)
        _,_,coupling=overlap_coupling(np.diag(np.exp(1j*np.array([.1,.3]))),.4)
        assert_allclose(np.diag(coupling),[.25j,.75j],atol=1e-12,rtol=0)
        with self.assertRaises(GaugeFailure):
            overlap_coupling(np.diag([1.,0.]),.4)
        with self.assertRaises(GaugeFailure):
            overlap_coupling(np.diag([-1.,1.]),.4)
        with self.assertRaises(GaugeFailure):
            dto_pair((OrbitalState((np.array([[1.],[0.],[0.]],complex),),([2],)),
                      OrbitalState((np.eye(3,dtype=complex)[:,1:],),([1,1],))),np.eye(3))


class FDTests(unittest.TestCase):
    def test_native_pyscf_signed_probe_matches_frozen_paper_axis(self):
        from pyscf import gto,scf
        from radiant.backends.pyscf import PySCFProvider
        from radiant.core.propagation import CNOptions
        fixture=json.loads((Path(__file__).parent/"fixtures/paper_signed_fd.json").read_text())
        mol=gto.M(atom=fixture["atom"],basis=fixture["basis"],unit=fixture["unit"],
                  charge=fixture["charge"],verbose=0)
        mf=scf.RHF(mol);mf.chkfile=None;mf.kernel()
        states=tuple(OrbitalState((np.array(c,dtype=complex),),(f,)) for c,f in
                     zip(fixture["coefficients_real"],fixture["occupations"]))
        execution=LocalExecution([PySCFProvider(mf,density_convention="paper_real"),
                                  PySCFProvider(mf,density_convention="paper_real")],states)
        probe=PaperFD(displacement=fixture["displacement"],dt=fixture["dt"],alpha=fixture["alpha"],
                      options=CNOptions(tolerance=fixture["tolerance"],max_iterations=fixture["max_iterations"]))
        result=probe.compute(execution)
        assert_allclose(result.real_direction,fixture["expected_direction"],atol=1e-12,rtol=0)
        self.assertEqual(execution.time,0.)
        nuclei=NuclearState(mol.atom_coords(),np.zeros((2,3)),[7294.,1836.])
        run=Trajectory(execution,nuclei,method=MASH([0.,0.,1.]))
        run.run(3,.05,n_substeps=2)
        assert_allclose([f["time"] for f in run.frames],np.arange(1,4)*.05,atol=0,rtol=0)
        self.assertTrue(np.isfinite(run.frames[-1]["total_energy"]))

    def test_signed_independent_probe_and_central_state_restore(self):
        ToyProvider.trace=[]
        run=LocalExecution([ToyProvider(0),ToyProvider(1)],alphas=[80.,40.],time=2.)
        before=run.snapshot()
        result=PaperFD().compute(run)
        assert_allclose(np.linalg.norm(result.real_direction),1.,atol=1e-12)
        self.assertTrue(np.iscomplexobj(result.raw))
        self.assertEqual(run.time,2.)
        self.assertEqual({round(t,12) for _,t,_,_ in ToyProvider.trace},{-2.,2.,6.})
        for provider in run.providers:
            self.assertIsNone(provider.cache)
        for a,b in zip(run.states,before.states):
            assert_allclose(a.density(),b.density(),atol=0)
        baseline=LocalExecution([ToyProvider(0),ToyProvider(1)],alphas=[1.,1.],time=2.)
        assert_allclose(PaperFD().compute(baseline).raw,result.raw,atol=1e-12,rtol=0)
        with self.assertRaises(ValueError):
            PaperFD(steps=2)

    def test_zero_axis_and_failed_branch_restore_all_states(self):
        run=LocalExecution([ToyProvider(0,rate=0.),ToyProvider(1,rate=0.)])
        with self.assertRaises(ZeroNACError):
            PaperFD().compute(run)
        self.assertEqual(run.time,0.)
        run=LocalExecution([ToyProvider(0),ToyProvider(1)])
        run.providers[1].failure_time=1.
        with self.assertRaisesRegex(RuntimeError,"Injected"):
            PaperFD().compute(run)
        self.assertEqual(run.time,0.)
        assert_allclose(run.positions,[[0,0,0]],atol=0)
        self.assertTrue(all(p.cache is None for p in run.providers))


class SpinTests(unittest.TestCase):
    def test_public_populations_and_legacy_order_conversion(self):
        for spin in ([0,0,1],[0,0,-1],[.3,.4,np.sqrt(.75)]):
            c=spin_to_coefficients(spin)
            assert_allclose(coefficients_to_spin(c),spin,atol=1e-12,rtol=0)
            assert_allclose(abs(c[1])**2,(1+spin[2])/2,atol=1e-12,rtol=0)
            method=MASH(np.array(spin))
            self.assertEqual(method.active_state,int(spin[2]>0))
        old=np.array([.6,0.,.8])
        public=legacy_to_public_spin(old,legacy_state_order=("ground","excited"))
        assert_allclose(public,[.6,0,-.8])
        self.assertEqual(MASH(public).active_state,0)
        with self.assertRaises(ValueError):
            legacy_to_public_spin(old,legacy_state_order=("unknown","unknown"))
        for c in (np.array([1e-9, np.sqrt(1-1e-18)*np.exp(.7j)]),
                  np.array([np.sqrt(1-1e-18), 1e-9*np.exp(-.4j)])):
            spin=coefficients_to_spin(c)
            assert_allclose(coefficients_to_spin(spin_to_coefficients(spin)),spin,atol=1e-12,rtol=0)

    def test_paper_generator_coefficients_and_first_crossing(self):
        c=np.array([np.sqrt(.9),1j*np.sqrt(.1)])
        method=MASH(coefficients_to_spin(c))
        reference=method.coefficients.copy()
        q=np.array([[np.cos(.2),-np.sin(.2)],[np.sin(.2),np.cos(.2)]],complex)
        crossings=[]
        for step in range(1,21):
            old=method.active_state
            method.propagate([0.,.3],[0.,.3],q,.4)
            reference=np.diag(np.exp(-.2j*np.array([0.,.3])))@q.conj().T@np.diag(np.exp(-.2j*np.array([0.,.3])))@reference
            assert_allclose(method.coefficients,reference,atol=1e-12,rtol=0)
            self.assertEqual(method.active_state,int(abs(reference[1])**2>abs(reference[0])**2))
            if method.active_state!=old:
                crossings.append(step)
        self.assertTrue(crossings)

    def test_allowed_frustrated_zero_projection_and_direction_sign(self):
        masses=np.array([2.,5.]);axis=np.array([[1.,0,0],[0,0,0]])
        for initial,p,expected in ((0,[[2.,0,0],[0,1,0]],"accepted"),
                                   (1,[[-2.,0,0],[0,1,0]],"accepted"),
                                   (1,[[0.,1,0],[0,2,0]],"accepted"),
                                   (0,[[0.,1,0],[0,2,0]],"frustrated")):
            results=[]
            for sign in (1,-1):
                method=MASH([.6,0.,.8 if initial==0 else -.8])
                before=method.spin.copy()
                momentum,outcome,_=method.resolve_hop(p,sign*axis,[0.,.3],masses,initial_state=initial)
                self.assertEqual(outcome,expected)
                final=1-initial if expected=="accepted" else initial
                self.assertEqual(method.active_state,final)
                assert_allclose(np.sum(momentum**2/masses[:,None])/2+[0.,.3][final],
                                np.sum(np.array(p)**2/masses[:,None])/2+[0.,.3][initial],atol=2e-15)
                assert_allclose(method.spin,before if expected=="accepted" else -before,atol=1e-12)
                results.append(momentum)
            assert_allclose(*results,atol=2e-15,rtol=0)


class DriverTests(unittest.TestCase):
    def hopping_run(self, initial=1, momentum=0.):
        method=MASH([.6,0.,.8 if initial else -.8])
        def propagate(*args):
            method.coefficients=spin_to_coefficients([.6,0.,-.8 if initial else .8])
        method.propagate=propagate
        nac=type("Axis",(),{"compute":lambda self,run:NACResult(np.ones((1,3),complex),np.array([[1.,0,0]]),1.)})()
        run=Trajectory(LocalExecution([ToyProvider(0,force=.1),ToyProvider(1,force=.2)]),
                       NuclearState([[0,0,0]],[[momentum,0,0]],[2.]),method=method,nac=nac)
        return run

    def test_hop_events_active_forces_and_second_half_kick_order(self):
        for initial,p,outcome in ((1,0.,"accepted"),(0,2.,"accepted"),(0,0.,"frustrated")):
            run=self.hopping_run(initial,p)
            frame=run.step(.4)
            event=run.hop_events[0]
            self.assertEqual(event[:5],(1,.4,initial,1-initial,outcome))
            force=.2 if initial else .1
            endpoint_p=p+.4*force
            assert_allclose(event[5],endpoint_p**2/4,atol=1e-15)
            self.assertEqual(run.active_state,1-initial if outcome=="accepted" else initial)
            self.assertAlmostEqual(frame["positions"][0,0],(p+.2*force)*.4/2)

    def test_zero_nac_saves_failure_and_rolls_back_spin_nuclei_and_electrons(self):
        for accepted in (0,1):
            run=self.hopping_run()
            if accepted:
                run.method.propagate=lambda *args:None
                run.step(.4)
                initial=run.method.active_state
                def cross(*args):
                    run.method.coefficients=spin_to_coefficients([.6,0.,-.8 if initial else .8])
                run.method.propagate=cross
            saved_nuclei=deepcopy(run.nuclei)
            saved_spin=run.method.spin.copy()
            saved_states=tuple(s.copy() for s in run.execution.states)
            def zero(*args):
                raise ZeroNACError("The NACV is zero")
            run.nac.compute=zero
            with tempfile.TemporaryDirectory() as folder:
                output=Path(folder)/"failed.npz"
                with self.assertRaises(ZeroNACError):
                    run.run(1,.4,output=output)
                with np.load(output,allow_pickle=False) as data:
                    self.assertFalse(data["completed"])
                    self.assertEqual(data["completed_steps"],accepted)
                    self.assertEqual(data["failed_step"],accepted+1)
                    self.assertEqual(len(data["time"]),accepted)
                    self.assertEqual(len(data["spin"]),accepted)
                    self.assertEqual(data["hop_events"][-1]["outcome"],"zero_nac")
                    self.assertEqual(data["hop_events"][-1]["time"],(accepted+1)*.4)
            assert_allclose(run.nuclei.positions,saved_nuclei.positions,atol=0)
            assert_allclose(run.nuclei.momenta,saved_nuclei.momenta,atol=0)
            assert_allclose(run.method.spin,saved_spin,atol=0)
            for state,saved in zip(run.execution.states,saved_states):
                assert_allclose(state.density(),saved.density(),atol=0)
            self.assertEqual(run.time,accepted*.4)
            self.assertEqual(run.execution.time,run.time)


if __name__ == "__main__":
    unittest.main()
