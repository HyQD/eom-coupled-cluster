import abc
from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.cc_helper import AmplitudeContainer


class TDEOMCC(TimeDependentCoupledCluster, metaclass=abc.ABCMeta):
    def __init__(self, system, t):
        super().__init__(system)
        self.t = t
        self.construct_hbar()

    @staticmethod
    def construct_amplitude_template(truncation, n, m, np):
        codes = {"S": 1, "D": 2, "T": 3, "Q": 4}
        levels = [codes[c] for c in truncation[2:]]

        # start with t_0 and l_0
        t = [np.array([0], dtype=np.complex128)]
        l = [np.array([0], dtype=np.complex128)]

        for lvl in levels:
            shape = lvl * [m] + lvl * [n]
            t.append(np.zeros(shape, dtype=np.complex128))
            l.append(np.zeros(shape[::-1], dtype=np.complex128))

        return AmplitudeContainer(t=t, l=l, np=np)

    @abc.abstractmethod
    def construct_hbar(self):
        pass

    def __call__(self, current_time, prev_amp):
        o, v = self.system.o, self.system.v

        prev_amp = self._amp_template.from_array(prev_amp)
        R_old, L_old = prev_amp

        self.update_hamiltonian(current_time, prev_amp)

        h = self.system.h
        u = self.system.u
        f = self.system.construct_fock_matrix(h, u)

        V_t = self.h
        if self.system._add_h_0:
            V_t = V_t - h

        t_new = [
            -1j
            * rhs_t_func(V_t, f, u, *R_old, *L_old, self.t, self.hbar, o, v, np=self.np)
            for rhs_t_func in self.rhs_t_amplitudes()
        ]

        l_new = [
            1j
            * rhs_l_func(V_t, f, u, *R_old, *L_old, self.t, self.hbar, o, v, np=self.np)
            for rhs_l_func in self.rhs_l_amplitudes()
        ]

        self.last_timestep = current_time

        return AmplitudeContainer(t=t_new, l=l_new, np=self.np).asarray()
