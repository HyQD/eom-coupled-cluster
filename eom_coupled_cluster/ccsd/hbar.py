from opt_einsum import contract


def build_Foo(f, u, t2, o, v):
    Foo = f[o, o].copy()
    Foo += 0.5 * contract("mnef, efjn->mj", u[o, o, v, v], t2)
    return Foo


def build_Fvv(f, u, t2, o, v):
    Fvv = f[v, v].copy()
    Fvv -= 0.5 * contract("bfmn, mnef->be", t2, u[o, o, v, v])
    return Fvv


def build_Wvvov(f, u, t2, o, v):
    Wvvov = u[v, v, o, v].copy()
    Wvvov -= contract("me, abjm->abje", f[o, v], t2)
    Pab = contract("afjm, bmef->abje", t2, u[v, o, v, v])
    Wvvov += Pab - Pab.swapaxes(0, 1)
    Wvvov += 0.5 * contract("abmn, mnje->abje", t2, u[o, o, o, v])
    return Wvvov


def build_Wvooo(f, u, t2, o, v):
    Wvooo = u[v, o, o, o].copy()
    Wvooo += contract("me, beij->bmij", f[o, v], t2)
    Wvooo += 0.5 * contract("efij, bmef->bmij", t2, u[v, o, v, v])
    Pij = contract("bein, mnje->bmij", t2, u[o, o, o, v])
    Wvooo += Pij - Pij.swapaxes(2, 3)
    return Wvooo


def build_Wvoov(f, u, t2, o, v):
    Wvoov = u[v, o, o, v].copy()
    Wvoov += contract("afin, mnef->amie", t2, u[o, o, v, v])
    return Wvoov


def build_Wovvo(f, u, t2, o, v):
    Wovvo = u[o, v, v, o].copy()
    Wovvo += contract("inaf, efmn->ieam", u[o, o, v, v], t2)
    return Wovvo


def build_Wvvvv(f, u, t2, o, v):
    Wvvvv = u[v, v, v, v].copy()
    Wvvvv += 0.5 * contract("abmn, mnef->abef", t2, u[o, o, v, v])
    return Wvvvv


def build_Woooo(f, u, t2, o, v):
    Woooo = u[o, o, o, o].copy()
    Woooo += 0.5 * contract("efij, mnef->mnij", t2, u[o, o, v, v])
    return Woooo


def build_Wovoo(f, u, t2, o, v):
    Wovoo = u[o, v, o, o].copy()
    Wovoo -= contract("me, beij->mbij", f[o, v], t2)
    Wovoo += 0.5 * contract("efij, mbef->mbij", t2, u[o, v, v, v])
    Wovoo += 2 * contract("mnie, bejn->mbij", u[o, o, o, v], t2)

    return Wovoo


def build_Wvvvo(f, u, t2, o, v):
    Wvvvo = u[v, v, v, o].copy()
    Wvvvo -= contract("me, abmi->abei", f[o, v], t2)
    Wvvvo += 0.5 * contract("mnei, abmn->abei", u[o, o, v, o], t2)

    Pab = contract("mbef, afmi->abei", u[o, v, v, v], t2)
    Wvvvo -= 2 * Pab  # - Pab.swapaxes(0, 1)

    return Wvvvo
