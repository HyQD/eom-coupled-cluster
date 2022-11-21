from opt_einsum import contract


def build_right_sigma_0(V_t, f, u, R0, R1, R2, L0, L1, L2, t, FW, o, v, np):
    t2 = t[0]

    sigma_0 = contract("em, me->", R1, f[o, v])
    sigma_0 += 0.25 * contract("efmn, mnef->", R2, u[o, o, v, v])
    # sigma_0 += e_corr * R0
    # sigma_0 += contract('mm->', V_t[o,o]) * R0

    # Contributions due to time-dependent one-body operator V_t(t)
    sigma_0 += contract("em, me->", R1, V_t[o, v])

    return sigma_0


def build_right_sigma_ai(V_t, f, u, R0, R1, R2, L0, L1, L2, t, FW, o, v, np):
    t2 = t[0]

    sigma_ai = contract("ei, ae->ai", R1, FW["vv"])
    sigma_ai -= contract("am, mi->ai", R1, FW["oo"])
    sigma_ai += contract("aeim, me->ai", R2, f[o, v])
    sigma_ai += contract("em, amie->ai", R1, FW["voov"])
    sigma_ai += 0.5 * contract("efim, amef->ai", R2, u[v, o, v, v])
    sigma_ai -= 0.5 * contract("aemn, mnie->ai", R2, u[o, o, o, v])

    # Contributions due to time-dependent one-body operator f(t)
    sigma_ai += contract("ei,ae->ai", R1, V_t[v, v])
    sigma_ai -= contract("am, mi->ai", R1, V_t[o, o])
    sigma_ai += contract("aeim, me->ai", R2, V_t[o, v])
    sigma_ai += R0 * contract("aeim, me->ai", t2, V_t[o, v])
    sigma_ai += R0 * V_t[v, o]
    # sigma_ai += contract('mm->', f[o,o]) * R1

    return sigma_ai


def build_right_sigma_abij(V_t, f, u, R0, R1, R2, L0, L1, L2, t, FW, o, v, np):
    t2 = t[0]

    sigma_abij = np.zeros(R2.shape, dtype=R2.dtype)

    Zoo = contract("em, mnje->nj", R1, u[o, o, o, v])
    Pij = contract("abin, nj->abij", t2, Zoo)
    sigma_abij += Pij - Pij.swapaxes(2, 3)

    Zvv = contract("em, bmef->bf", R1, u[v, o, v, v])
    Pab = contract("afij, bf->abij", t2, Zvv)
    sigma_abij -= Pab - Pab.swapaxes(0, 1)

    Pij = contract("ei, abje->abij", R1, FW["vvov"])
    sigma_abij -= Pij - Pij.swapaxes(2, 3)

    Pab = contract("am, bmij->abij", R1, FW["vooo"])
    sigma_abij += Pab - Pab.swapaxes(0, 1)

    Pab = contract("aeij, be->abij", R2, FW["vv"])
    sigma_abij += Pab - Pab.swapaxes(0, 1)

    Pij = contract("abim, mj->abij", R2, FW["oo"])
    sigma_abij -= Pij - Pij.swapaxes(2, 3)

    sigma_abij += 0.5 * contract("abmn, mnij->abij", R2, FW["oooo"])
    sigma_abij += 0.5 * contract("efij, abef->abij", R2, FW["vvvv"])
    Pabij = contract("aeim, bmje->abij", R2, FW["voov"])
    sigma_abij += (
        Pabij
        - Pabij.swapaxes(0, 1)
        - Pabij.swapaxes(2, 3)
        + Pabij.swapaxes(0, 1).swapaxes(2, 3)
    )

    Goo = contract("efim, mnef->ni", R2, u[o, o, v, v])
    Pij = contract("abjn, ni->abij", t2, Goo)
    sigma_abij -= 0.5 * (Pij - Pij.swapaxes(2, 3))

    Gvv = contract("aemn, mnef->af", R2, u[o, o, v, v])
    Pab = contract("bfij, af->abij", t2, Gvv)
    sigma_abij -= 0.5 * (Pab - Pab.swapaxes(0, 1))

    # Contributions due to time-dependent one-body operator f(t)
    Pij = contract("ei, abjm, me->abij", R1, t2, V_t[o, v])
    sigma_abij += Pij - Pij.swapaxes(2, 3)

    Pab = contract("am, beij, me->abij", R1, t2, V_t[o, v])
    sigma_abij += Pab - Pab.swapaxes(0, 1)

    Pab = contract("aeij, be->abij", R2, V_t[v, v])
    sigma_abij += Pab - Pab.swapaxes(0, 1)

    Pij = contract("abim, mj->abij", R2, V_t[o, o])
    sigma_abij -= Pij - Pij.swapaxes(2, 3)

    Pab = contract("aeij, be->abij", t2, V_t[v, v])
    sigma_abij += R0 * (Pab - Pab.swapaxes(0, 1))

    Pij = contract("abim, mj->abij", t2, V_t[o, o])
    sigma_abij -= R0 * (Pij - Pij.swapaxes(2, 3))

    Pabij = contract("ai, bejm, me->abij", R1, t2, V_t[o, v])
    Pabij += contract("ai, bj->abij", R1, V_t[v, o])
    sigma_abij += (
        Pabij
        - Pabij.swapaxes(0, 1)
        - Pabij.swapaxes(2, 3)
        + Pabij.swapaxes(0, 1).swapaxes(2, 3)
    )

    # sigma_abij += contract('mm->', V_t[o,o]) * R2

    return sigma_abij
