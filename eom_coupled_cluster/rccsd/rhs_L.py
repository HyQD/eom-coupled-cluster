from opt_einsum import contract


def build_left_sigma_0(V_t, f, u, R0, R1, R2, L0, L1, L2, t, FW, o, v, np):
    t2 = t[0]
    sigma_0 = contract("ia, ai->", L1, V_t[v, o])
    sigma_0 += 2 * contract("ia, jb, abij->", L1, V_t[o, v], t2)
    sigma_0 -= contract("ia, jb, abji->", L1, V_t[o, v], t2)
    sigma_0 -= contract("ki, ijab, bajk->", V_t[o, o], L2, t2)
    sigma_0 += contract("ac, ijab, bcji->", V_t[v, v], L2, t2)
    return sigma_0


def build_left_sigma_ia(V_t, f, u, R0, R1, R2, L0, L1, L2, t, Hbar, o, v, np):
    t2 = t[0]
    # l1 equations
    sigma_ia = 2.0 * Hbar["ov"] * L0
    sigma_ia += contract("ie,ea->ia", L1, Hbar["vv"])
    sigma_ia -= contract("im,ma->ia", Hbar["oo"], L1)
    sigma_ia += 2 * contract("ieam,me->ia", Hbar["ovvo"], L1)
    sigma_ia -= contract("iema,me->ia", Hbar["ovov"], L1)
    sigma_ia += contract("imef,efam->ia", L2, Hbar["vvvo"])
    sigma_ia -= contract("iemn,mnae->ia", Hbar["ovoo"], L2)
    sigma_ia -= 2 * contract("eifa,ef->ia", Hbar["vovv"], build_Gvv(t2, L2, np))
    sigma_ia += contract("eiaf,ef->ia", Hbar["vovv"], build_Gvv(t2, L2, np))
    sigma_ia -= 2 * contract("mina,mn->ia", Hbar["ooov"], build_Goo(t2, L2, np))
    sigma_ia += contract("imna,mn->ia", Hbar["ooov"], build_Goo(t2, L2, np))

    # Contributions due to time-dependent one-body operator V_t(t)
    sigma_ia += 2 * L0 * V_t[o, v]
    sigma_ia += contract("ib,ba->ia", L1, V_t[v, v])
    sigma_ia -= contract("ic,jkab,bckj->ia", V_t[o, v], L2, t2)
    sigma_ia -= contract("ka,ijbc,bckj->ia", V_t[o, v], L2, t2)
    sigma_ia -= contract("kc,ijab,bckj->ia", V_t[o, v], L2, t2)
    sigma_ia += 2 * contract("kc,ijab,bcjk->ia", V_t[o, v], L2, t2)
    sigma_ia -= contract("ja,ij->ia", L1, V_t[o, o])
    sigma_ia += contract("bj,ijab->ia", V_t[v, o], L2)
    # sigma_ia += 2 * contract("ia,jj->ia", L1, V_t)
    return sigma_ia


def build_left_sigma_ijab(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t, Hbar, o, v, np
):
    t2 = t[0]
    tmp = u[o, o, v, v].copy()
    Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)

    sigma_ijab = Loovv * L0
    sigma_ijab += 2 * contract("ia,jb->ijab", L1, Hbar["ov"])
    sigma_ijab -= contract("ja,ib->ijab", L1, Hbar["ov"])
    sigma_ijab += contract("ijeb,ea->ijab", L2, Hbar["vv"])
    sigma_ijab -= contract("im,mjab->ijab", Hbar["oo"], L2)
    sigma_ijab += 0.5 * contract("ijmn,mnab->ijab", Hbar["oooo"], L2)

    ###########################################################################
    # Avoid explicit construction og Hvvvv
    sigma_ijab += 0.5 * contract("ijef, efab->ijab", L2, u[v, v, v, v])

    tmp_ijmn = contract("ijef, efmn->ijmn", L2, t2)
    sigma_ijab += 0.5 * contract("ijmn, mnab->ijab", tmp_ijmn, u[o, o, v, v])
    ###########################################################################

    sigma_ijab += 2 * contract("ie,ejab->ijab", L1, Hbar["vovv"])
    sigma_ijab -= contract("ie,ejba->ijab", L1, Hbar["vovv"])
    sigma_ijab -= 2 * contract("mb,jima->ijab", L1, Hbar["ooov"])
    sigma_ijab += contract("mb,ijma->ijab", L1, Hbar["ooov"])
    sigma_ijab += 2 * contract("ieam,mjeb->ijab", Hbar["ovvo"], L2)
    sigma_ijab -= contract("iema,mjeb->ijab", Hbar["ovov"], L2)
    sigma_ijab -= contract("mibe,jema->ijab", L2, Hbar["ovov"])
    sigma_ijab -= contract("mieb,jeam->ijab", L2, Hbar["ovvo"])
    sigma_ijab += contract("ijeb,ae->ijab", Loovv, build_Gvv(t2, L2, np))
    sigma_ijab -= contract("mi,mjab->ijab", build_Goo(t2, L2, np), Loovv)

    sigma_ijab += sigma_ijab.swapaxes(0, 1).swapaxes(2, 3)

    # Contributions due to time-dependent one-body operator V_t(t)
    sigma_ijab -= contract("ib,ja->ijab", L1, V_t[o, v])
    sigma_ijab -= contract("ja,ib->ijab", L1, V_t[o, v])
    sigma_ijab += 2 * contract("ia,jb->ijab", L1, V_t[o, v])
    sigma_ijab += 2 * contract("jb,ia->ijab", L1, V_t[o, v])
    sigma_ijab += contract("ca,ijcb->ijab", V_t[v, v], L2)
    sigma_ijab += contract("cb,ijac->ijab", V_t[v, v], L2)
    sigma_ijab -= contract("ik,kjab->ijab", V_t[o, o], L2)
    sigma_ijab -= contract("jk,ikab->ijab", V_t[o, o], L2)
    # sigma_ijab += 2 * contract("kk,ijab->ijab", V_t, L2)
    return sigma_ijab


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += contract("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= contract("ijab,ebij->ae", l2, t2)
    return Gvv
