from opt_einsum import contract


def compute_one_body_density_matrix(R0, R1, R2, L0, L1, L2, t2, o, v, np, out=None):
    """
    rho^q_p = <tilde{Psi}|a_p^\dagger a_q|Psi>
    """

    nocc = o.stop
    nvirt = v.stop - nocc

    if out is None:
        out = np.zeros((nocc + nvirt, nocc + nvirt), dtype=R2.dtype)

    identity_occ = np.eye(nocc)
    tmp = (
        L0 * R0 + contract("me, em->", L1, R1) + 0.25 * contract("mnef, efmn->", L2, R2)
    )
    out[o, o] = tmp * identity_occ
    out[o, o] -= 0.5 * R0 * contract("jmef, efim->ji", L2, t2)
    out[o, o] -= contract("je, ei->ji", L1, R1)
    out[o, o] -= 0.5 * contract("jmef, efim->ji", L2, R2)

    out[o, v] = R0 * L1
    out[o, v] += contract("imae, em->ia", L2, R1)

    out[v, o] = L0 * R1
    out[v, o] += R0 * contract("me, aeim->ai", L1, t2)
    out[v, o] += contract("me, aeim->ai", L1, R2)
    out[v, o] += contract("mnef, em, afin->ai", L2, R1, t2)
    out[v, o] -= 0.5 * contract("mnef, ei, afmn->ai", L2, R1, t2)
    out[v, o] -= 0.5 * contract("mnef, am, efin->ai", L2, R1, t2)

    out[v, v] = 0.5 * R0 * contract("mnae, bemn->ba", L2, t2)
    out[v, v] += contract("ma, bm->ba", L1, R1)
    out[v, v] += 0.5 * contract("mnae, bemn->ba", L2, R2)

    return out
