from opt_einsum import contract


def build_left_sigma_ia(V_t, f, u, R0, R1, R2, L0, L1, L2, t2, Hbar, o, v, np):

    # l1 equations
    r_l1 = 2.0 * Hbar["ov"] * L0
    r_l1 += contract("ie,ea->ia", L1, Hbar["vv"])
    r_l1 -= contract("im,ma->ia", Hbar["oo"], L1)
    r_l1 += 2 * contract("ieam,me->ia", Hbar["ovvo"], L1)
    r_l1 -= contract("iema,me->ia", Hbar["ovov"], L1)
    r_l1 += contract("imef,efam->ia", L2, Hbar["vvvo"])
    r_l1 -= contract("iemn,mnae->ia", Hbar["ovoo"], L2)
    r_l1 -= 2 * contract("eifa,ef->ia", Hbar["vovv"], build_Gvv(t2, L2, np))
    r_l1 += contract("eiaf,ef->ia", Hbar["vovv"], build_Gvv(t2, L2, np))
    r_l1 -= 2 * contract("mina,mn->ia", Hbar["ooov"], build_Goo(t2, L2, np))
    r_l1 += contract("imna,mn->ia", Hbar["ooov"], build_Goo(t2, L2, np))

    return r_l1


def build_left_sigma_ijab(
    V_t, f, u, R0, R1, R2, L0, L1, L2, t2, Hbar, o, v, np
):

    tmp = u[o, o, v, v].copy()
    Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)

    r_l2 = Loovv * L0
    r_l2 += 2 * contract("ia,jb->ijab", L1, Hbar["ov"])
    r_l2 -= contract("ja,ib->ijab", L1, Hbar["ov"])
    r_l2 += contract("ijeb,ea->ijab", L2, Hbar["vv"])
    r_l2 -= contract("im,mjab->ijab", Hbar["oo"], L2)
    r_l2 += 0.5 * contract("ijmn,mnab->ijab", Hbar["oooo"], L2)

    ###########################################################################
    # Avoid explicit construction og Hvvvv
    r_l2 += 0.5 * contract("ijef, efab->ijab", L2, u[v, v, v, v])

    tmp_ijmn = contract("ijef, efmn->ijmn", L2, t2)
    r_l2 += 0.5 * contract("ijmn, mnab->ijab", tmp_ijmn, u[o, o, v, v])
    ###########################################################################

    r_l2 += 2 * contract("ie,ejab->ijab", L1, Hbar["vovv"])
    r_l2 -= contract("ie,ejba->ijab", L1, Hbar["vovv"])
    r_l2 -= 2 * contract("mb,jima->ijab", L1, Hbar["ooov"])
    r_l2 += contract("mb,ijma->ijab", L1, Hbar["ooov"])
    r_l2 += 2 * contract("ieam,mjeb->ijab", Hbar["ovvo"], L2)
    r_l2 -= contract("iema,mjeb->ijab", Hbar["ovov"], L2)
    r_l2 -= contract("mibe,jema->ijab", L2, Hbar["ovov"])
    r_l2 -= contract("mieb,jeam->ijab", L2, Hbar["ovvo"])
    r_l2 += contract("ijeb,ae->ijab", Loovv, build_Gvv(t2, L2, np))
    r_l2 -= contract("mi,mjab->ijab", build_Goo(t2, L2, np), Loovv)

    # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
    r_l2 += r_l2.swapaxes(0, 1).swapaxes(2, 3)
    return r_l2


def build_Goo(t2, l2, np):
    Goo = 0
    Goo += contract("abmj,ijab->mi", t2, l2)
    return Goo


def build_Gvv(t2, l2, np):
    Gvv = 0
    Gvv -= contract("ijab,ebij->ae", l2, t2)
    return Gvv
