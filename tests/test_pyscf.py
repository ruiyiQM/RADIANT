"""Native PySCF and independent fixed-X energy differences; PySCF is required."""

from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto, scf

from radiant.backends.pyscf import PySCFProvider
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


if __name__ == "__main__":
    unittest.main()
