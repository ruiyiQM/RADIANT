"""Native PySCF and independent fixed-X energy differences; PySCF is required."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto, scf

from radiant.backends.pyscf import PySCFProvider
from radiant.backends.pyscf_nac import ZVectorNAC
from radiant.core.execution import LocalExecution
from radiant.core.propagation import CNOptions, crank_nicolson
from radiant.core.state import OrbitalState
from radiant.core.transport import lowdin, transport


def mean_field(kind, *, open_shell=False):
    mol = gto.M(atom="H 0 0 0" if open_shell else "H 0 0 0; H .2 0 1.4",
                basis="6-31g", unit="Bohr", spin=int(open_shell), verbose=0)
    mf = {"rhf":scf.RHF,"rks":dft.RKS,"uhf":scf.UHF,"uks":dft.UKS}[kind](mol)
    if kind.endswith("ks"):
        mf.xc = "b3lyp"
        mf.grids.level = 1
        mf.grids.prune = None
    mf.chkfile = None
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.kernel()
    return mf


class PySCFTests(unittest.TestCase):
    def test_native_complex_fock_energy_and_fixed_x_forces(self):
        for kind in ("rhf", "rks", "uhf", "uks"):
            with self.subTest(kind=kind):
                provider = PySCFProvider(mean_field(kind))
                inverse, _ = lowdin(provider.overlap)
                rng = np.random.default_rng(42)
                q = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
                c = inverse@q[:, :1]
                state = OrbitalState((c, c.conj()) if provider.unrestricted else (c,),
                    ([1], [1]) if provider.unrestricted else ([2],))
                dm = state.density()
                native = provider.mf.get_hcore()+provider.mf.get_veff(dm=dm)
                assert_allclose(provider.fock(dm),native,atol=1e-12,rtol=0)
                self.assertAlmostEqual(provider.energy(state),provider.mf.energy_tot(dm=dm),places=12)
                force = provider.forces(state)
                origin, overlap = provider.positions, provider.overlap
                numerical = np.zeros_like(force)
                for ia in range(len(origin)):
                    for axis in range(3):
                        energies=[]
                        for sign in (1, -1):
                            trial=deepcopy(provider)
                            positions=origin.copy()
                            positions[ia,axis]+=sign*1e-4
                            trial.prepare_geometry(positions)
                            energies.append(trial.energy(transport(state,overlap,trial.overlap)))
                        numerical[ia,axis]=-(energies[0]-energies[1])/2e-4
                assert_allclose(force,numerical,atol=2e-7,rtol=0)

    def test_paper_real_is_explicit_and_differs_for_complex_density(self):
        mf=mean_field("rhf")
        native=PySCFProvider(mf)
        paper=PySCFProvider(mf,density_convention="paper_real")
        inv,_=lowdin(native.overlap)
        c=inv@np.array([[1.],[.5j],[.2],[.1j]])
        c/=np.linalg.norm(lowdin(native.overlap)[1]@c)
        state=OrbitalState((c,),([2],))
        dm=state.density()
        assert_allclose(paper.fock(dm),mf.get_hcore()+mf.get_veff(dm=dm.real),atol=1e-12,rtol=0)
        self.assertGreater(np.linalg.norm(native.fock(dm)-paper.fock(dm)),1e-3)
        origin, overlap = paper.positions, paper.overlap
        energies=[]
        for sign in (1,-1):
            trial=deepcopy(paper)
            positions=origin.copy(); positions[0,2]+=sign*1e-4
            trial.prepare_geometry(positions)
            energies.append(trial.energy(transport(state,overlap,trial.overlap)))
        self.assertAlmostEqual(paper.forces(state)[0,2],-(energies[0]-energies[1])/2e-4,places=7)

    def test_stationary_gradient_and_empty_spin(self):
        for kind in ("rhf", "rks", "uhf", "uks"):
            mf=mean_field(kind,open_shell=kind.startswith("u"))
            provider=PySCFProvider(mf)
            state=provider.initial_state()
            grad=mf.nuc_grad_method()
            if kind.endswith("ks"):
                grad.grid_response=True
            assert_allclose(provider.forces(state),-grad.kernel(),atol=2e-7,rtol=0)
            if provider.unrestricted:
                self.assertEqual(state.coefficients[1].shape[1],0)
            result, diagnostics=crank_nicolson(state,provider.overlap,provider.fock,0.,.1,
                                               options=CNOptions(tolerance=1e-11))
            assert_allclose(result.density(),state.density(),atol=1e-9,rtol=0)
            self.assertLess(diagnostics.residual,1e-11)

    def test_geometry_preserves_settings_and_unsupported_models_fail(self):
        mf=mean_field("rks")
        provider=PySCFProvider(mf)
        positions=provider.positions.copy(); positions[1,2]+=.01
        provider.prepare_geometry(positions)
        self.assertEqual(provider.mf.xc,mf.xc)
        self.assertEqual(provider.mf.grids.level,1)
        self.assertIsNone(provider.mf.grids.prune)
        assert_allclose(mf.mol.atom_coords(),[[0,0,0],[.2,0,1.4]],atol=1e-15)
        mf.converged=False
        with self.assertRaises(ValueError):
            PySCFProvider(mf)
        mf.converged=True
        with self.assertRaises(ValueError):
            PySCFProvider(mf.density_fit())

    def test_range_separated_complex_forces_and_unequal_spins(self):
        for unrestricted in (False,True):
            mol=gto.M(atom="He 0 0 0; H .2 0 1.4",basis="6-31g",unit="Bohr",
                      charge=0 if unrestricted else 1,spin=int(unrestricted),verbose=0)
            mf=(dft.UKS if unrestricted else dft.RKS)(mol,xc="wb97x")
            mf.grids.level=1;mf.grids.prune=None;mf.chkfile=None;mf.kernel()
            provider=PySCFProvider(mf)
            inverse,_=lowdin(provider.overlap)
            rng=np.random.default_rng(23)
            q=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
            state=OrbitalState((inverse@q[:,:2],inverse@q[:,2:3]) if unrestricted else (inverse@q[:,:1],),
                               ([1,1],[1]) if unrestricted else ([2],))
            force=provider.forces(state)
            original,overlap=provider.positions,provider.overlap
            for axis in (0,2):
                energies=[]
                for sign in (1,-1):
                    trial=deepcopy(provider);positions=original.copy();positions[0,axis]+=sign*1e-4
                    trial.prepare_geometry(positions)
                    energies.append(trial.energy(transport(state,overlap,trial.overlap)))
                assert_allclose(force[0,axis],-(energies[0]-energies[1])/2e-4,atol=2e-7,rtol=0)


class ZVectorNACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).parent/"fixtures/table1_recomputed.npz",allow_pickle=False) as fixture:
            cls.fixture=dict(fixture)
        mol=gto.M(atom=list(zip(("N","N","H","H"),cls.fixture["positions_bohr"])),
                  basis="6-31g*",unit="Bohr",verbose=0)
        cls.mf=scf.RHF(mol);cls.mf.chkfile=None;cls.mf.kernel()
        cls.states=tuple(OrbitalState((cls.fixture[name+"_coeff"],),(cls.fixture[name+"_occ"],))
                         for name in ("ground","excited"))

    def execution(self,states,convention="paper_real"):
        return LocalExecution([PySCFProvider(self.mf,density_convention=convention) for _ in states],states)

    def test_table1_static_direction_and_state_phases(self):
        result=ZVectorNAC().compute(self.execution(self.states))
        self.assertGreater(np.sum(result.real_direction*self.fixture["static_direction"]),1-1e-7)
        rng=np.random.default_rng(7)
        phased=tuple(OrbitalState((s.coefficients[0]*np.exp(2j*np.pi*rng.random(s.coefficients[0].shape[1])),),
                                  s.occupations) for s in self.states)
        raw=ZVectorNAC().compute(self.execution(phased)).raw
        phase=np.vdot(result.raw,raw)/np.vdot(result.raw,result.raw)
        assert_allclose(abs(phase),1.,atol=1e-10,rtol=0)
        assert_allclose(raw,phase*result.raw,atol=1e-10*np.linalg.norm(raw),rtol=0)

    def test_energy_must_depend_on_real_density_only(self):
        with self.assertRaises(ValueError):
            ZVectorNAC().compute(self.execution(self.states,"complex"))
        with self.assertRaises(ValueError):
            ZVectorNAC(tolerance=0.)


if __name__ == "__main__":
    unittest.main()


class UnrestrictedPairTests(unittest.TestCase):
    """NO2 X 2A1 / A 2B2 as two UKS determinants (beta HOMO b2 -> LUMO a1) at a C2v geometry."""

    @classmethod
    def setUpClass(cls):
        half = np.radians(60)
        mol = gto.M(atom=f"N 0 0 0; O {1.2*np.sin(half)} {1.2*np.cos(half)} 0; O {-1.2*np.sin(half)} {1.2*np.cos(half)} 0",
                    basis="sto-3g", spin=1, verbose=0)

        def uks():
            mf = dft.UKS(mol); mf.xc = "pbe"; mf.grids.level = 1; mf.chkfile = None; mf.conv_tol = 1e-12
            return mf
        ground = uks(); ground.kernel()
        occ = np.array(ground.mo_occ, dtype=float)
        nb = int(occ[1].sum())
        occ[1, nb-1], occ[1, nb] = 0, 1
        excited = scf.addons.mom_occ(uks(), ground.mo_coeff, occ)
        excited.kernel(dm0=excited.make_rdm1(ground.mo_coeff, occ))
        assert ground.converged and excited.converged and np.array_equal(excited.mo_occ, occ)
        cls.mfs = (ground, excited)

    def execution(self):
        return LocalExecution([PySCFProvider(mf) for mf in self.mfs])

    def test_paper_fd_coupling_has_the_b2_symmetry(self):
        from radiant.core.coupling import PaperFD
        run = self.execution()
        now = run.snapshot()
        gram = run.overlap(now, now)
        assert_allclose(gram, np.eye(2), atol=1e-6, rtol=0)  # A1 and B2 determinants are orthogonal
        # The imaginary part (~1% here) is CN-tolerance noise: 1e-6 over a 0.02 bohr FD interval.
        raw = PaperFD().compute(run).raw.real
        self.assertGreater(np.linalg.norm(raw), 1e-2)
        # C2(y) swaps the O atoms and maps (x, y, z) -> (-x, y, -z); A1 x B2 is odd under it
        assert_allclose(raw[[0, 2, 1]], raw*[1, -1, 1], atol=1e-5*np.linalg.norm(raw), rtol=0)
        assert_allclose(raw[:, 2], 0, atol=1e-5*np.linalg.norm(raw), rtol=0)

    def test_gauge_alignment_and_rejected_pairs(self):
        from radiant.core.overlap import dto_pair
        run = self.execution()
        central = run.gauge
        rng = np.random.default_rng(11)
        phased = tuple(OrbitalState(tuple(c*np.exp(2j*np.pi*rng.random(c.shape[1])) for c in s.coefficients),
                                    s.occupations) for s in run.states)
        overlap = run.providers[0].overlap
        for a, b in zip(dto_pair(phased, overlap, reference=central, cross=overlap, align_all=True), central):
            for ca, cb in zip(a.coefficients, b.coefficients):
                assert_allclose(ca, cb, atol=1e-12, rtol=0)
        ground, excited = run.states
        fewer = OrbitalState((excited.coefficients[0][:, :-1], excited.coefficients[1]),
                             (excited.occupations[0][:-1], excited.occupations[1]))
        restricted = OrbitalState((ground.coefficients[0],), (2*ground.occupations[0],))
        for pair in ((ground, fewer), (restricted, excited)):
            with self.assertRaises(ValueError):
                dto_pair(pair, overlap)

    def test_zvector_nac_matches_static_fd_for_a_non_orthogonal_pair(self):
        from radiant.core.overlap import pair_overlap
        half = np.radians(55)
        mol = gto.M(atom=f"N 0 0 0; O {1.18*np.sin(half)} {1.18*np.cos(half)} 0; O {-1.24*np.sin(half)} {1.24*np.cos(half)} 0",
                    basis="sto-3g", spin=1, verbose=0)

        def mom(m, coeff, occ):
            mf = dft.UKS(m); mf.xc = "pbe"; mf.chkfile = None; mf.conv_tol = 1e-13
            if coeff is None:
                mf.kernel()
                return mf
            mf = scf.addons.mom_occ(mf, coeff, occ)
            mf.kernel(dm0=mf.make_rdm1(coeff, occ))
            assert mf.converged
            return mf
        ground = mom(mol, None, None)
        occ = np.array(ground.mo_occ, dtype=float)
        nb = int(occ[1].sum())
        occ[1, nb-1], occ[1, nb] = 0, 1
        pair = (ground, mom(mol, ground.mo_coeff, occ))
        run = LocalExecution([PySCFProvider(mf) for mf in pair])
        self.assertGreater(abs(run.overlap(run.snapshot(), run.snapshot())[0, 1]), 1e-2)
        d = ZVectorNAC().compute(run).raw
        assert_allclose(d.imag, 0, atol=1e-10, rtol=0)
        rng = np.random.default_rng(3)
        h = 1e-3
        for v in [d.real] + list(rng.normal(size=(2, 3, 3))):
            v = v/np.linalg.norm(v)
            ends = []
            for sign in (1, -1):
                m = mol.set_geom_(mol.atom_coords() + sign*h*v, unit="Bohr", inplace=False)
                moved = LocalExecution([PySCFProvider(mom(m, mf.mo_coeff, mf.mo_occ)) for mf in pair]).gauge
                o = pair_overlap(run.gauge, moved, gto.intor_cross("int1e_ovlp", mol, m))
                ends.append(o*np.sign(np.diag(o).real))
            fd = (ends[0] - ends[1])/(2*h)
            # FD truncation (h**2) limits the agreement to ~1e-4 |d|
            self.assertLess(abs(np.sum(d.real*v) - .5*(fd[0, 1] - fd[1, 0]).real), 5e-4*np.linalg.norm(d))
