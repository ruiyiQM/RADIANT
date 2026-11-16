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
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm

from radiant.core.coupling import NACResult, PaperFD, ZeroNACError
from radiant.core.execution import LocalExecution
from radiant.core.overlap import GaugeFailure, dto_pair, overlap_coupling, pair_overlap
from radiant.core.propagation import CNOptions
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
    def test_selected_recomputed_table1_direction_benchmark(self):
        from pyscf import gto,scf
        from radiant.backends.pyscf import PySCFProvider
        from radiant.core.propagation import CNOptions
        with np.load(Path(__file__).parent/"fixtures/table1_recomputed.npz",allow_pickle=False) as fixture:
            mol=gto.M(atom=list(zip(("N","N","H","H"),fixture["positions_bohr"])),
                      basis="6-31g*",unit="Bohr",verbose=0)
            mf=scf.RHF(mol);mf.chkfile=None;mf.kernel()
            assert_allclose(mf.get_ovlp(),fixture["overlap"],atol=1e-12,rtol=0)
            states=tuple(OrbitalState((fixture[name+"_coeff"],),(fixture[name+"_occ"],))
                         for name in ("ground","excited"))
            execution=LocalExecution([PySCFProvider(mf,density_convention="paper_real") for _ in states],states)
            for dt,expected in zip(fixture["dt_au"],fixture["absolute_dot"]):
                result=PaperFD(dt=float(dt),options=CNOptions(tolerance=1e-6,max_iterations=40),
                               subtract_drift=False).compute(execution)
                dot=np.sum(result.real_direction*fixture["static_direction"])
                self.assertGreater(dot,0.)
                assert_allclose(dot,expected,atol=1e-12,rtol=0)
                self.assertEqual(execution.time,0.)

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
        probe=PaperFD(displacement=fixture["displacement"],dt=fixture["dt"],alpha=fixture["alpha"],subtract_drift=False,
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
        with self.assertRaises(ZeroNACError):  # the toy coupling is pure time evolution: all drift
            PaperFD().compute(run)
        result=PaperFD(subtract_drift=False).compute(run)
        assert_allclose(np.linalg.norm(result.real_direction),1.,atol=1e-12)
        self.assertTrue(np.iscomplexobj(result.raw))
        self.assertEqual(run.time,2.)
        self.assertEqual({round(t,12) for _,t,_,_ in ToyProvider.trace},{-2.,2.,6.})
        for provider in run.providers:
            self.assertIsNone(provider.cache)
        for a,b in zip(run.states,before.states):
            assert_allclose(a.density(),b.density(),atol=0)
        baseline=LocalExecution([ToyProvider(0),ToyProvider(1)],alphas=[1.,1.],time=2.)
        assert_allclose(PaperFD(subtract_drift=False).compute(baseline).raw,result.raw,atol=1e-12,rtol=0)
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
        generator=np.array([[.03j,-1+.25j],[1+.25j,-.02j]])
        overlap=expm(.2*generator)@np.diag([.94,.98])
        crossings=[]
        for step in range(1,21):
            old=method.active_state
            method.propagate([0.,.3],[0.,.3],overlap,.4)
            u,_,vh=np.linalg.svd(overlap)
            reference=(expm(-.2j*np.diag([0.,.3]))@expm(-logm(u@vh))
                       @expm(-.2j*np.diag([0.,.3]))@reference)
            assert_allclose(method.coefficients,reference,atol=1e-12,rtol=0)
            cross=np.conj(reference[0])*reference[1]
            legacy_spin=np.array([2*cross.real,2*cross.imag,abs(reference[0])**2-abs(reference[1])**2])
            assert_allclose(method.spin,legacy_spin*[1,-1,-1],atol=1e-12,rtol=0)
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

    def test_complex_nac_uses_the_phase_independent_mass_weighted_axis(self):
        masses=np.array([2.,5.]);p=np.array([[2.,1,0],[0,1,3]])
        nac=np.array([[1.,.2,0],[0,.5,-.3]])+1j*np.array([[.3,-.4,.1],[.6,0,.2]])
        a,b=(part/np.sqrt(masses[:,None]) for part in (nac.real,nac.imag))
        phase=.5*np.arctan2(-2*np.sum(a*b),np.sum(a*a)-np.sum(b*b))  # maximises |Re(exp(i phase) nac)|
        expected=MASH([.6,0.,.8]).resolve_hop(p,(np.exp(1j*phase)*nac).real,[0.,.3],masses,initial_state=0)[0]
        for theta in (0.,1.,2.5,-2.):
            momentum,outcome,_=MASH([.6,0.,.8]).resolve_hop(p,np.exp(1j*theta)*nac,[0.,.3],masses,initial_state=0)
            self.assertEqual(outcome,"accepted")
            assert_allclose(momentum,expected,atol=1e-12,rtol=0)


class DriverTests(unittest.TestCase):
    def test_accepted_hop_completes_new_force_and_force_failure_rolls_back(self):
        original = ToyProvider.forces
        calls = []
        def counted(provider, state):
            calls.append(provider.branch)
            return original(provider, state)
        with patch.object(ToyProvider, 'forces', counted):
            run = self.hopping_run()
            run.step(.4)
            self.assertEqual(calls, [1, 1, 0])
            self.assertEqual(run.active_state, 0)
            assert_allclose(run.observation.forces[0], [[.1,0,0]])
        run = self.hopping_run()
        before = deepcopy(run.nuclei)
        spin = run.method.spin.copy()
        def failed(provider, state):
            if provider.branch == 0:
                raise RuntimeError('New active force failed')
            return original(provider, state)
        with patch.object(ToyProvider, 'forces', failed), self.assertRaisesRegex(RuntimeError, 'active force'):
            run.step(.4)
        self.assertEqual(run.time, 0.)
        self.assertEqual(run.active_state, 1)
        self.assertEqual(len(run.frames), 0)
        self.assertEqual(run.hop_events[-1][4], 'failed')
        assert_allclose(run.nuclei.positions, before.positions, atol=0, rtol=0)
        assert_allclose(run.method.spin, spin, atol=0, rtol=0)

    def hopping_run(self, initial=1, momentum=0., variant="cn_accelerated"):
        method=MASH([.6,0.,.8 if initial else -.8])
        def propagate(*args):
            method.coefficients=spin_to_coefficients([.6,0.,-.8 if initial else .8])
        method.propagate=propagate
        nac=type("Axis",(),{"compute":lambda self,run:NACResult(np.array([[1.,0,0]],complex),np.array([[1.,0,0]]),1.)})()
        run=Trajectory(LocalExecution([ToyProvider(0,force=.1),ToyProvider(1,force=.2)],
                                     options=CNOptions(variant)),
                       NuclearState([[0,0,0]],[[momentum,0,0]],[2.]),method=method,nac=nac)
        return run

    def test_hop_events_active_forces_and_second_half_kick_order(self):
        for variant, (initial,p,outcome) in product(("cn_accelerated", "pt_cn", "pt_im"),
                                                   ((1,0.,"accepted"),(0,2.,"accepted"),(0,0.,"frustrated"))):
            run=self.hopping_run(initial,p,variant)
            frame=run.step(.4)
            event=run.hop_events[0]
            self.assertEqual(event[:5],(1,.4,initial,1-initial,outcome))
            force=.2 if initial else .1
            endpoint_p=p+.4*force
            assert_allclose(event[5],endpoint_p**2/4,atol=1e-15)
            self.assertEqual(run.active_state,1-initial if outcome=="accepted" else initial)
            self.assertAlmostEqual(frame["positions"][0,0],(p+.2*force)*.4/2)

    def test_zero_nac_saves_failure_and_rolls_back_spin_nuclei_and_electrons(self):
        for variant, accepted in product(("cn_accelerated", "pt_cn", "pt_im"), (0,1)):
            run=self.hopping_run(variant=variant)
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


class TullyExecution:
    """Adiabatic double of an execution: Tully's simple avoided crossing along x (mass 2000)."""

    def __init__(self, x):
        self.x, self.time = float(x), 0.

    @staticmethod
    def adiabatic(x):
        a, b, c, d = .01, 1.6, .005, 1.
        v11, d11 = np.sign(x)*a*(1-np.exp(-b*abs(x))), a*b*np.exp(-b*abs(x))
        v12 = c*np.exp(-d*x*x)
        w = np.hypot(v11, v12)
        dw = (v11*d11 - 2*d*x*v12*v12)/w
        theta = .5*np.arctan2(v12, v11)
        forces = np.zeros((2, 1, 3)); forces[:, 0, 0] = [dw, -dw]
        return np.array([-w, w]), forces, np.array([[-np.sin(theta), np.cos(theta)], [np.cos(theta), np.sin(theta)]])

    @property
    def positions(self):
        return np.array([[self.x, 0., 0.]])

    def observe(self, *, forces=True):
        energies, force, _ = self.adiabatic(self.x)
        if not isinstance(forces, bool):
            force[[i for i in range(2) if i not in forces]] = np.nan
        return SimpleNamespace(time=self.time, energies=energies, forces=force)

    def snapshot(self):
        return (self.x, self.time)

    def restore(self, snapshot):
        self.x, self.time = snapshot

    def advance(self, end, dt, *, midpoint=None, n_substeps=1, forces=True):
        old = self.adiabatic(self.x)[2]
        self.x, self.time = float(end[0][0]), self.time + dt
        return SimpleNamespace(observation=self.observe(forces=forces), overlap_raw=old.T @ self.adiabatic(self.x)[2])


class ReversibleHopTests(unittest.TestCase):
    axis = SimpleNamespace(compute=lambda execution: SimpleNamespace(raw=np.array([[1., 0, 0]])))

    def test_selected_forces_match_all_forces_through_localized_hops(self):
        class Eager(TullyExecution):
            def observe(self, *, forces=True):
                return super().observe(forces=True)
        for p,spin,steps in ((16.16,(.6,.2,-.77),20),(6.,(.3,.95,-.1),30)):
            spin=np.asarray(spin)/np.linalg.norm(spin)
            runs=[Trajectory(kind(-1.5), NuclearState([[-1.5,0,0]],[[p,0,0]],[2000.]),
                             method=MASH(spin),nac=self.axis,hop_tolerance=1e-8)
                  for kind in (TullyExecution,Eager)]
            for run in runs:run.run(steps,20.)
            self.assertTrue(runs[0].hop_events)
            self.assertEqual(runs[0].hop_events,runs[1].hop_events)
            for a,b in zip(runs[0].frames,runs[1].frames):
                for key in a:assert_allclose(a[key],b[key],atol=0,rtol=0)

    def tully(self, dt, steps, tolerance, x=-1.5, p=16.16, spin=(.6, .2, -.77)):
        spin = np.asarray(spin)/np.linalg.norm(spin)
        run = Trajectory(TullyExecution(x), NuclearState([[x, 0, 0]], [[p, 0, 0]], [2000.]), method=MASH(spin),
                         nac=self.axis, hop_tolerance=tolerance)
        run.run(steps, dt)
        return run

    def retrace_error(self, forward, dt, steps, tolerance, x, p, spin):
        back = self.tully(dt, steps, tolerance, forward.nuclei.positions[0, 0], -forward.nuclei.momenta[0, 0],
                          forward.method.spin*[1, -1, 1])
        return max(abs(back.nuclei.positions[0, 0]-x), abs(back.nuclei.momenta[0, 0]+p),
                   np.abs(back.method.spin*[1, -1, 1]-np.asarray(spin)/np.linalg.norm(spin)).max())

    def test_accepted_hop_is_located_inside_the_step_and_retraced(self):
        for tolerance in (1e-8, None):
            forward = self.tully(20., 20, tolerance)
            self.assertEqual([e[4] for e in forward.hop_events], ["accepted"])
            error = self.retrace_error(forward, 20., 20, tolerance, -1.5, 16.16, (.6, .2, -.77))
            if tolerance is None:   # paper scheme: hop at the step end, not reversible
                self.assertEqual(forward.hop_events[0][1] % 20., 0.)
                self.assertGreater(error, 1e-3)
            else:
                self.assertGreater(abs(forward.hop_events[0][1]/20.-round(forward.hop_events[0][1]/20.)), 1e-3)
                self.assertLess(error, 1e-8)

    def test_rev_pc_is_second_order_through_an_accepted_hop(self):
        reference = self.tully(.25, 1600, 1e-10)
        errors = []
        for dt in (10., 5., 2.5):
            run = self.tully(dt, int(400/dt), 1e-8)
            self.assertEqual(len(run.hop_events), 1)
            errors.append(max(abs(run.nuclei.positions[0, 0]-reference.nuclei.positions[0, 0]),
                              np.abs(run.method.spin-reference.method.spin).max()))
        self.assertTrue(all(ratio > 3.5 for ratio in np.array(errors[:-1])/errors[1:]), errors)

    def test_frustrated_pi_hops_retrace_and_tolerance_is_validated(self):
        forward = self.tully(20., 30, 1e-8, p=6., spin=(.3, .95, -.1))
        self.assertEqual({e[4] for e in forward.hop_events}, {"frustrated"})
        self.assertLess(self.retrace_error(forward, 20., 30, 1e-8, -1.5, 6., (.3, .95, -.1)), 1e-8)
        nuclei = NuclearState([[0, 0, 0]], [[0, 0, 0]], [2000.])
        for method, tolerance in ((None, 1e-4), (MASH([0, 0, 1.]), 0.), (MASH([0, 0, 1.]), 1.), (MASH([0, 0, 1.]), np.nan)):
            with self.assertRaises(ValueError):
                Trajectory(TullyExecution(0.), nuclei, method=method, nac=self.axis, hop_tolerance=tolerance)


if __name__ == "__main__":
    unittest.main()
