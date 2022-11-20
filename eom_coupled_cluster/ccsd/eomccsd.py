from coupled_cluster import CCSD
from coupled_cluster.cc_helper import (
    AmplitudeContainer,
)

from opt_einsum import contract


class EOMCCSD(CCSD):
    def compute_ground_state(
        self, t_args=[], t_kwargs={}, l_args=[], l_kwargs={}, t1_transform=True
    ):
        super().compute_ground_state(
            t_args=t_args, t_kwargs=t_kwargs, l_args=l_args, l_kwargs=l_kwargs
        )

        np = self.np
        self.R0 = np.complex128(1 + 0j)
        self.R1 = np.zeros(self.t_1.shape, dtype=np.complex128)
        self.R2 = np.zeros(self.t_2.shape, dtype=np.complex128)
        self.L0 = np.complex128(1 + 0j)
        self.L1 = np.complex128(self.l_1)
        self.L2 = np.complex128(self.l_2)

        if t1_transform:
            from scipy.linalg import expm

            T1 = np.zeros((self.system.l, self.system.l))
            T1[self.system.v, self.system.o] += self.t_1.real
            C = expm(T1)
            C_tilde = expm(-T1)
            self.system.change_basis(C=C, C_tilde=C_tilde)

    def get_amplitudes(self):
        return AmplitudeContainer(
            t=[self.R0, self.R1, self.R2], l=[self.L0, self.L1, self.L2], np=self.np
        )
