from opt_einsum import contract


def build_left_sigma_0(V_t, f, u, R0, R1, R2, L0, L1, L2, t2, FW, o, v, np):

    sigma_0 = contract("me, em->", L1, V_t[v, o])
    sigma_0 += contract("nf, efmn, me->", L1, t2, V_t[o, v])

    sigma_0 += 0.5 * contract("mneg, fgmn, ef->", L2, t2, V_t[v, v])
    sigma_0 -= 0.5 * contract("moef, efno, nm->", L2, t2, V_t[o, o])

    return sigma_0


def build_left_sigma_ia(V_t, f, u, R0, R1, R2, L0, L1, L2, t2, FW, o, v, np):

    sigma_ia = L0 * f[o, v]
    sigma_ia += contract("ie, ea->ia", L1, FW["vv"])
    sigma_ia -= contract("ma, im->ia", L1, FW["oo"])
    sigma_ia += contract("me, ieam->ia", L1, FW["ovvo"])

    sigma_ia += 0.5 * contract("imef, efam->ia", L2, FW["vvvo"])
    sigma_ia -= 0.5 * contract("mnae, iemn->ia", L2, FW["ovoo"])

    tmp_oo = contract("mnef, efmo->no", L2, t2)
    sigma_ia -= 0.5 * contract("no, ioan->ia", tmp_oo, u[o, o, v, o])

    tmp_vv = contract("mnef, egmn->gf", L2, t2)
    sigma_ia += 0.5 * contract("gf, ifag->ia", tmp_vv, u[o, v, v, v])

    # Contributions due to time-dependent one-body operator V_t(t)
    sigma_ia += L0 * V_t[o, v]
    sigma_ia -= contract("ma, im->ia", L1, V_t[o, o])
    sigma_ia += contract("ie, ea->ia", L1, V_t[v, v])

    tmp_vv = contract("mnaf, efmn->ea", L2, t2)
    sigma_ia -= 0.5 * contract("ea, ie->ia", tmp_vv, V_t[o, v])

    tmp_oo = contract("inef, efmn->im", L2, t2)
    sigma_ia -= 0.5 * contract("im, ma->ia", tmp_oo, V_t[o, v])

    tmp_ai = contract("efmn, me->fn", t2, V_t[o, v])
    sigma_ia += contract("inaf,fn->ia", L2, tmp_ai)
    sigma_ia += contract("imae,em->ia", L2, V_t[v, o])

    return sigma_ia


def build_left_sigma_ijab(V_t, f, u, R0, R1, R2, L0, L1, L2, t2, FW, o, v, np):

    sigma_ijab = L0 * u[o, o, v, v]

    Pabij = contract("ia, jb->ijab", f[o, v], L1)
    sigma_ijab += (
        Pabij
        - Pabij.swapaxes(0, 1)
        - Pabij.swapaxes(2, 3)
        + Pabij.swapaxes(0, 1).swapaxes(2, 3)
    )

    Pab = contract("ma, ijbm->ijab", L1, u[o, o, v, o])
    sigma_ijab += Pab - Pab.swapaxes(2, 3)

    Pij = contract("ie, jeab->ijab", L1, u[o, v, v, v])
    sigma_ijab -= Pij - Pij.swapaxes(0, 1)

    Pab = contract("ijae, eb->ijab", L2, FW["vv"])
    sigma_ijab += Pab - Pab.swapaxes(2, 3)

    Pij = contract("imab, jm->ijab", L2, FW["oo"])
    sigma_ijab -= Pij - Pij.swapaxes(0, 1)

    Pabij = contract("imae, jebm->ijab", L2, FW["ovvo"])
    sigma_ijab += (
        Pabij
        - Pabij.swapaxes(0, 1)
        - Pabij.swapaxes(2, 3)
        + Pabij.swapaxes(0, 1).swapaxes(2, 3)
    )

    sigma_ijab += 0.5 * contract("ijef, efab->ijab", L2, FW["vvvv"])
    sigma_ijab += 0.5 * contract("mnab, ijmn->ijab", L2, FW["oooo"])

    tmp_vv = contract("mnae, efmn->fa", L2, t2)
    Pab = 0.5 * contract("fa, ijbf->ijab", tmp_vv, u[o, o, v, v])
    sigma_ijab -= Pab - Pab.swapaxes(2, 3)

    tmp_oo = contract("imef, efmn->in", L2, t2)
    Pij = 0.5 * contract("in, jnab->ijab", tmp_oo, u[o, o, v, v])
    sigma_ijab -= Pij - Pij.swapaxes(0, 1)

    # Contributions due to time-dependent one-body operator V_t(t)
    Pabij = contract("ia, jb->ijab", L1, V_t[o, v])
    sigma_ijab += (
        Pabij
        - Pabij.swapaxes(0, 1)
        - Pabij.swapaxes(2, 3)
        + Pabij.swapaxes(0, 1).swapaxes(2, 3)
    )

    Pij = contract("imab, jm->ijab", L2, V_t[o, o])
    sigma_ijab -= Pij - Pij.swapaxes(0, 1)

    Pab = contract("ijae, eb->ijab", L2, V_t[v, v])
    sigma_ijab += Pab - Pab.swapaxes(2, 3)

    return sigma_ijab
