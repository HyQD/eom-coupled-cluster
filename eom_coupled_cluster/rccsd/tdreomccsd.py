import abc
from eom_coupled_cluster.tdeomcc import TDEOMCC
from eom_coupled_cluster.rccsd.density_matrices import compute_one_body_density_matrix

from eom_coupled_cluster.rccsd.hbar import *
from eom_coupled_cluster.rccsd.rhs_R import (
    build_right_sigma_0,
    build_right_sigma_ai,
    build_right_sigma_abij,
)
from eom_coupled_cluster.rccsd.rhs_L import (
    build_left_sigma_0,
    build_left_sigma_ia,
    build_left_sigma_ijab,
)


class TDREOMCCSD(TDEOMCC):
    truncation = "CCSD"

    def construct_hbar(self):
        o = self.system.o
        v = self.system.v
        t2 = self.t[0]
        f = self.system.construct_fock_matrix(self.h, self.u)
        u = self.u
        np = self.np

        Loovv = build_Loovv(u, o, v, np)
        Looov = build_Looov(u, o, v, np)
        Lvovv = build_Lvovv(u, o, v, np)

        self.hbar = {
            "oo": build_Foo(f, Looov, Loovv, t2, o, v, np),
            "ov": build_Fov(f, Loovv, o, v, np),
            "vv": build_Fvv(f, Lvovv, Loovv, t2, o, v, np),
            "vovv": build_Hvovv(u, t2, o, v, np),
            "ooov": build_Hooov(u, t2, o, v, np),
            "ovov": build_Hovov(u, t2, o, v, np),
            "ovvo": build_Hovvo(u, Loovv, t2, o, v, np),
            "vvvv": build_Hvvvv(u, t2, o, v, np),
            "oooo": build_Hoooo(u, t2, o, v, np),
            "vvvo": build_Hvvvo(f, u, Loovv, Lvovv, t2, o, v, np),
            "ovoo": build_Hovoo(f, u, Loovv, Looov, t2, o, v, np),
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
