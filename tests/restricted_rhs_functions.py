import numpy as np
from eom_coupled_cluster.rccsd.rhs_R import (
    build_right_sigma_ai_drudge,
    build_right_sigma_abij_drudge,
    build_right_sigma_0,
    build_right_sigma_ai,
    build_right_sigma_abij,
)
from eom_coupled_cluster.rccsd.rhs_L import (
    build_left_sigma_ia,
    build_left_sigma_ijab,
)
from eom_coupled_cluster.rccsd.hbar import *

nv = 20
no = 6
l = nv + no

o = slice(0, no)
v = slice(no, l)
u = np.random.random((l, l, l, l))
f = np.random.random((l, l))
V_t = np.random.random((l, l))

t2 = np.random.random((nv, nv, no, no))

R0 = np.random.random()
R1 = np.random.random((nv, no))
R2 = np.random.random((nv, nv, no, no))

L0 = np.random.random()
L1 = np.random.random((no, nv))
L2 = np.random.random((no, no, nv, nv))

# Enforce t^{ab}_{ij} = t^{ba}_{ji}
u = 0.5 * (u + u.swapaxes(0, 1).swapaxes(2, 3))
t2 = 0.5 * (t2 + t2.swapaxes(0, 1).swapaxes(2, 3))
R2 = 0.5 * (R2 + R2.swapaxes(0, 1).swapaxes(2, 3))
L2 = 0.5 * (L2 + L2.swapaxes(0, 1).swapaxes(2, 3))

test_r1 = build_right_sigma_ai_drudge(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, o, v, np
)
test_r2 = build_right_sigma_abij_drudge(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, o, v, np
)

Loovv = build_Loovv(u, o, v, np)
Lvovv = build_Lvovv(u, o, v, np)
Looov = build_Looov(u, o, v, np)

hbar = {
    "oo": build_Foo(f, Looov, Loovv, t2, o, v, np),
    "ov": build_Fov(f, Loovv, o, v, np),
    "vv": build_Fvv(f, Lvovv, Loovv, t2, o, v, np),
    "ovvo": build_Hovvo(u, Loovv, t2, o, v, np),
    "ovov": build_Hovov(u, t2, o, v, np),
    "vvvo": build_Hvvvo(f, u, Loovv, Lvovv, t2, o, v, np),
    "ovoo": build_Hovoo(f, u, Loovv, Looov, t2, o, v, np),
    "vovv": build_Hvovv(u, t2, o, v, np),
    "ooov": build_Hooov(u, t2, o, v, np),
    "oooo": build_Hoooo(u, t2, o, v, np),
    "vvvv": build_Hvvvv(u, t2, o, v, np),
}

r0 = build_right_sigma_0(V_t, f, u, R0, R1, R2, L0, L1, L2, t2, hbar, o, v, np)
r1_factorized = build_right_sigma_ai(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, hbar, o, v, np
)
r2_factorized = build_right_sigma_abij(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, hbar, o, v, np
)

l1_factorized = build_left_sigma_ia(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, hbar, o, v, np
)
l2_factorized = build_left_sigma_ijab(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, hbar, o, v, np
)

print(r0)
print(np.allclose(r1_factorized, test_r1))
print(np.allclose(r2_factorized, test_r2))
