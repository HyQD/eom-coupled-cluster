from coupled_cluster.cc_helper import AmplitudeContainer

from eom_coupled_cluster.tdeomcc import TDEOMCC
from eom_coupled_cluster.ccsd.density_matrices import compute_one_body_density_matrix
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

    def construct_hbar(self):
        o = self.system.o
        v = self.system.v
        t2 = self.t[0]
        f = self.system.construct_fock_matrix(self.h, self.u)
        u = self.u

        self.hbar = {
            "oo": build_Foo(f, u, t2, o, v),
            "vv": build_Fvv(f, u, t2, o, v),
            "vvov": build_Wvvov(f, u, t2, o, v),
            "vooo": build_Wvooo(f, u, t2, o, v),
            "voov": build_Wvoov(f, u, t2, o, v),
            "ovvo": build_Wovvo(f, u, t2, o, v),
            "vvvv": build_Wvvvv(f, u, t2, o, v),
            "oooo": build_Woooo(f, u, t2, o, v),
            "vvvo": build_Wvvvo(f, u, t2, o, v),
            "ovoo": build_Wovoo(f, u, t2, o, v),
        }

    def compute_one_body_density_matrix(self, current_time, y):
        R0, R1, R2, L0, L1, L2 = self._amp_template.from_array(y).unpack()
        t2 = self.t[0]
        return compute_one_body_density_matrix(
            R0, R1, R2, L0, L1, L2, t2, self.o, self.v, np=self.np
        )

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
