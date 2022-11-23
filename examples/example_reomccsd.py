import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from eom_coupled_cluster import REOMCCSD, TDREOMCCSD

from scipy.integrate import complex_ode
import matplotlib.pyplot as plt

from rk4_integrator import Rk4Integrator


class sine_square_laser:
    def __init__(self, F_str, omega, tprime, phase=0):
        self.F_str = F_str
        self.omega = omega
        self.tprime = tprime
        self.phase = phase

    def __call__(self, t):
        pulse = (
            (np.sin(np.pi * t / self.tprime) ** 2)
            * np.heaviside(t, 1.0)
            * np.heaviside(self.tprime - t, 1.0)
            * np.sin(self.omega * t + self.phase)
            * self.F_str
        )
        return pulse


molecule = "He 0.0 0.0 0.0"

system = construct_pyscf_system_rhf(
    molecule=molecule,
    basis="cc-pvdz",
    add_spin=False,
    anti_symmetrize=False,
    charge=0,
)


reomccsd = REOMCCSD(system, verbose=True)


ground_state_tolerance = 1e-10
reomccsd.compute_ground_state(
    t_kwargs=dict(tol=ground_state_tolerance),
    l_kwargs=dict(tol=ground_state_tolerance),
    t1_transform=True,
)


y0 = reomccsd.get_amplitudes().asarray()

tdreomccsd = TDREOMCCSD(system, [reomccsd.t_2])

r = complex_ode(tdreomccsd).set_integrator("Rk4Integrator", dt=0.01)
r.set_initial_value(y0, 0)


polarization = np.zeros(3)
polarization[2] = 1
system.set_time_evolution_operator(
    DipoleFieldInteraction(
        sine_square_laser(F_str=1.0, omega=2.87, tprime=5, phase=np.pi / 2),
        polarization_vector=polarization,
    )
)

dt = 0.01
T = 10
num_steps = int(T // dt) + 1
time_points = []


import tqdm

time = []
R0L0 = []
dipole_moment = []
z = reomccsd.system.dipole_moment[2]

tdreomccsd.system._add_h_0 = False
aa0 = tdreomccsd.system.h_t(4)
tdreomccsd.system._add_h_0 = True
aa1 = tdreomccsd.system.h_t(4)

aa1 -= tdreomccsd.system.h

print(np.max(np.abs(aa0 - aa1)))

for n in tqdm.tqdm(range(num_steps)):
    time.append(r.t)
    rho_qp = tdreomccsd.compute_one_body_density_matrix(r.t, r.y)
    dipole_moment.append(
        tdreomccsd.compute_one_body_expectation_value(r.t, r.y, -z, make_hermitian=False)
    )
    r.integrate(r.t + dt)
    R0, R1, R2, L0, L1, L2 = tdreomccsd._amp_template.from_array(r.y).unpack()
    R0L0.append(np.abs(L0.conj() * R0) ** 2)
    tdreomccsd.system.h_t(r.t)

output = {
    "time_points": np.array(time),
    "dipole_moment": np.array(dipole_moment),
}

np.savez("d_t_restricted", **output)

plt.plot(time, dipole_moment)
# plt.plot(time,R0L0)
plt.show()
