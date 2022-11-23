from opt_einsum import contract


def compute_one_body_density_matrix(
    R0, R1, R2, L0, L1, L2, t2, o, v, np, out=None
):
    """
    rho^q_p = <tilde{Psi}|E^p_q|Psi>
    """

    nocc = o.stop
    nvirt = v.stop - nocc

    if out is None:
        out = np.zeros((nocc + nvirt, nocc + nvirt), dtype=R2.dtype)

    identity_occ = np.eye(nocc)

    out[o, o] += 2 * L0 * R0 * identity_occ
    out[o, o] += 2 * identity_occ * contract("ka,ak->", L1, R1)
    out[o, o] += identity_occ * contract("klab,abkl->", L2, R2)
    out[o, o] -= contract("ja,ai->ji", L1, R1)
    out[o, o] -= contract("kjab,baik->ji", L2, R2)
    out[o, o] -= R0 * contract("kjab,baik->ji", L2, t2)

    out[v, o] += 2 * L0 * R1
    out[v, o] -= contract("aj,jkbc,bcik->ai", R1, L2, t2)
    out[v, o] -= contract("bi,jkbc,acjk->ai", R1, L2, t2)
    out[v, o] -= contract("bj,jkbc,acki->ai", R1, L2, t2)
    out[v, o] += 2 * contract("bj,jkbc,acik->ai", R1, L2, t2)
    out[v, o] -= contract("jb,abji->ai", L1, R2)
    out[v, o] += 2 * contract("jb,abij->ai", L1, R2)
    out[v, o] -= R0 * contract("jb,abji->ai", L1, t2)
    out[v, o] += 2 * R0 * contract("jb,abij->ai", L1, t2)

    out[o, v] += R0 * L1
    out[o, v] += contract("bj,ijab->ia", R1, L2)

    out[v, v] += contract("ijac,bcij->ba", L2, R2)
    out[v, v] += R0 * contract("ijac,bcij->ba", L2, t2)
    out[v, v] += contract("ia,bi->ba", L1, R1)

    return out
