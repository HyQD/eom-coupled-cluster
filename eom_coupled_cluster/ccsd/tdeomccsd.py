from coupled_cluster.cc_helper import AmplitudeContainer

from eom_coupled_cluster.tdeomcc import TDEOMCC
from eom_coupled_cluster.ccsd.hbar import *
from eom_coupled_cluster.ccsd.rhs_R import (
    build_right_sigma_0,
    build_right_sigma_ai,
    build_right_sigma_abij,
)
from eom_coupled_cluster.ccsd.rhs_L import (
    build_left_sigma_0,
    build_left_sigma_ia,
    build_left_sigma_ijab,
)


class TDEOMCCSD(TDEOMCC):
    truncation = "CCSD"

    def construct_FW(self):
        o = self.system.o
        v = self.system.v
        t2 = self.t2.real
        f = self.system.construct_fock_matrix(self.h, self.u)
        u = self.u

        self.FW = {
            "oo": build_Foo(f, u, t2.real, o, v),
            "vv": build_Fvv(f, u, t2.real, o, v),
            "vvov": build_Wvvov(f, u, t2.real, o, v),
            "vooo": build_Wvooo(f, u, t2.real, o, v),
            "voov": build_Wvoov(f, u, t2.real, o, v),
            "ovvo": build_Wovvo(f, u, t2.real, o, v),
            "vvvv": build_Wvvvv(f, u, t2.real, o, v),
            "oooo": build_Woooo(f, u, t2.real, o, v),
            "vvvo": build_Wvvvo(f, u, t2.real, o, v),
            "ovoo": build_Wovoo(f, u, t2.real, o, v),
        }

    def compute_one_body_density_matrix(self, current_time, y):
        R0, R1, R2, L0, L1, L2 = self._amp_template.from_array(y).unpack()

        """
        rho^q_p = <tilde{Psi}|a_p^\dagger a_q|Psi>
        """

        np = self.np

        o = self.o
        v = self.v
        t2 = self.t2

        nocc = self.o.stop
        nvirt = self.v.stop - nocc
        rho = np.zeros((nocc + nvirt, nocc + nvirt), dtype=R2.dtype)

        identity_occ = np.eye(nocc)
        tmp = (
            L0 * R0
            + contract("me, em->", L1, R1)
            + 0.25 * contract("mnef, efmn->", L2, R2)
        )
        rho[o, o] = tmp * identity_occ
        rho[o, o] -= 0.5 * R0 * contract("jmef, efim->ji", L2, t2)
        rho[o, o] -= contract("je, ei->ji", L1, R1)
        rho[o, o] -= 0.5 * contract("jmef, efim->ji", L2, R2)

        rho[o, v] = R0 * L1
        rho[o, v] += contract("imae, em->ia", L2, R1)

        rho[v, o] = L0 * R1
        rho[v, o] += R0 * contract("me, aeim->ai", L1, t2)
        rho[v, o] += contract("me, aeim->ai", L1, R2)
        rho[v, o] += contract("mnef, em, afin->ai", L2, R1, t2)
        rho[v, o] -= 0.5 * contract("mnef, ei, afmn->ai", L2, R1, t2)
        rho[v, o] -= 0.5 * contract("mnef, am, efin->ai", L2, R1, t2)

        rho[v, v] = 0.5 * R0 * contract("mnae, bemn->ba", L2, t2)
        rho[v, v] += contract("ma, bm->ba", L1, R1)
        rho[v, v] += 0.5 * contract("mnae, bemn->ba", L2, R2)

        return rho

    def rhs_t_amplitudes(self):
        yield build_right_sigma_0
        yield build_right_sigma_ai
        yield build_right_sigma_abij

    def rhs_l_amplitudes(self):
        yield build_left_sigma_0
        yield build_left_sigma_ia
        yield build_left_sigma_ijab

    def compute_energy(self, current_time, y):
        pass

    def compute_left_reference_overlap(self, current_time, y):
        pass

    def compute_overlap(self, current_time, y_a, y_b):
        pass

    def compute_two_body_density_matrix(self, current_time, y):
        pass

    def rhs_t_0_amplitude(self, *args, **kwargs):
        pass
