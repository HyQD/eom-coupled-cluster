"""
Copyright (c) 2014-2018, The Psi4NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the Psi4NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Modified from the original source code: 
    https://github.com/psi4/psi4numpy/blob/cbef6ddcb32ccfbf773befea6dc4aaae2b428776/Coupled-Cluster/RHF/helper_cchbar.py
"""

from opt_einsum import contract


def build_Loovv(u, o, v, np):
    tmp = u[o, o, v, v].copy()
    Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
    return Loovv


def build_Looov(u, o, v, np):
    tmp = u[o, o, o, v].copy()
    Looov = 2.0 * tmp - tmp.swapaxes(0, 1)
    return Looov


def build_Lvovv(u, o, v, np):
    tmp = u[v, o, v, v].copy()
    Lvovv = 2.0 * tmp - tmp.swapaxes(2, 3)
    return Lvovv


# F and W are the one and two body intermediates which appear in the CCSD
# T1 and T2 equations. Please refer to helper_ccenergy file for more details.


def build_Fov(f, Loovv, o, v, np):
    """<m|Hbar|e> = F_me = f_me + t_nf <mn||ef>"""
    Hov = f[o, v].copy()
    return Hov


def build_Foo(f, Looov, Loovv, t2, o, v, np):
    """
    <m|Hbar|i> = F_mi + 0.5 * t_ie F_me = f_mi + t_ie f_me
                 + t_ne <mn||ie> + tau_inef <mn||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hoo = np.zeros((nocc, nocc), dtype=t2.dtype)

    Hoo += f[o, o]
    Hoo += contract("efin,mnef->mi", t2, Loovv)
    return Hoo


def build_Fvv(f, Lvovv, Loovv, t2, o, v, np):
    """
    <a|Hbar|e> = F_ae - 0.5 * t_ma F_me = f_ae - t_ma f_me
                 + t_mf <am||ef> - tau_mnfa <mn||fe>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvv = np.zeros((nvirt, nvirt), dtype=t2.dtype)

    Hvv = f[v, v].copy()
    Hvv -= contract("famn,mnfe->ae", t2, Loovv)
    return Hvv


def build_Hoooo(u, t2, o, v, np):
    """
    <mn|Hbar|ij> = W_mnij + 0.25 * tau_ijef <mn||ef> = <mn||ij>
                   + P(ij) t_je <mn||ie> + 0.5 * tau_ijef <mn||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hoooo = np.zeros((nocc, nocc, nocc, nocc), dtype=t2.dtype)

    Hoooo += u[o, o, o, o]
    Hoooo += contract("efij,mnef->mnij", t2, u[o, o, v, v])
    return Hoooo


def build_Hvvvv(u, t2, o, v, np):
    """
    <ab|Hbar|ef> = W_abef + 0.25 * tau_mnab <mn||ef> = <ab||ef>
                   - P(ab) t_mb <am||ef> + 0.5 * tau_mnab <mn||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvvvv = np.zeros((nvirt, nvirt, nvirt, nvirt), dtype=t2.dtype)

    Hvvvv += u[v, v, v, v]
    Hvvvv += contract("abmn,mnef->abef", t2, u[o, o, v, v])
    return Hvvvv


def build_Hvovv(u, t2, o, v, np):
    """<am|Hbar|ef> = <am||ef> - t_na <nm||ef>"""
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvovv = np.zeros((nvirt, nocc, nvirt, nvirt), dtype=t2.dtype)

    Hvovv += u[v, o, v, v]
    return Hvovv


def build_Hooov(u, t2, o, v, np):
    """<mn|Hbar|ie> = <mn||ie> + t_if <mn||fe>"""
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hooov = np.zeros((nocc, nocc, nocc, nvirt), dtype=t2.dtype)

    Hooov += u[o, o, o, v]
    return Hooov


def build_Hovvo(u, Loovv, t2, o, v, np):
    """
    <mb|Hbar|ej> = W_mbej - 0.5 * t_jnfb <mn||ef> = <mb||ej> + t_jf <mb||ef>
                   - t_nb <mn||ej> - (t_jnfb + t_jf t_nb) <nm||fe>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovvo = np.zeros((nocc, nvirt, nvirt, nocc), dtype=t2.dtype)

    Hovvo += u[o, v, v, o]
    Hovvo -= contract("fbjn,nmfe->mbej", t2, u[o, o, v, v])
    Hovvo += contract("bfjn,nmfe->mbej", t2, Loovv)
    return Hovvo


def build_Hovov(u, t2, o, v, np):
    """
    <mb|Hbar|je> = - <mb|Hbar|ej> = <mb||je> + t_jf <bm||ef> - t_nb <mn||je>
                   - (t_jnfb + t_jf t_nb) <nm||ef>
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovov = np.zeros((nocc, nvirt, nocc, nvirt), dtype=t2.dtype)

    Hovov += u[o, v, o, v]
    Hovov -= contract("fbjn,nmef->mbje", t2, u[o, o, v, v])
    return Hovov


def build_Hvvvo(f, u, Loovv, Lvovv, t2, o, v, np):
    """
    <ab|Hbar|ei> = <ab||ei> - F_me t_miab + t_if Wabef + 0.5 * tau_mnab <mn||ei>
                   - P(ab) t_miaf <mb||ef> - P(ab) t_ma {<mb||ei> - t_nibf <mn||ef>}
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hvvvo = np.zeros((nvirt, nvirt, nvirt, nocc), dtype=t2.dtype)

    # <ab||ei>

    Hvvvo += u[v, v, v, o]

    # - Fme t_miab

    Hvvvo -= contract("me,abmi->abei", f[o, v], t2)

    # t_if Wabef
    # 0.5 * tau_mnab <mn||ei>

    Hvvvo += contract("abmn,mnei->abei", t2, u[o, o, v, o])

    # - P(ab) t_miaf <mb||ef>
    Hvvvo -= contract("faim,mbef->abei", t2, u[o, v, v, v])
    Hvvvo -= contract("fbim,amef->abei", t2, u[v, o, v, v])
    Hvvvo += contract("fbmi,amef->abei", t2, Lvovv)
    return Hvvvo


def build_Hovoo(f, u, Loovv, Looov, t2, o, v, np):
    """
    <mb|Hbar|ij> = <mb||ij> - Fme t_ijbe - t_nb Wmnij + 0.5 * tau_ijef <mb||ef>
                   + P(ij) t_jnbe <mn||ie> + P(ij) t_ie {<mb||ej> - t_njbf <mn||ef>}
    """
    nocc = t2.shape[2]
    nvirt = t2.shape[0]
    Hovoo = np.zeros((nocc, nvirt, nocc, nocc), dtype=t2.dtype)

    # <mb||ij>

    Hovoo += u[o, v, o, o]

    # - Fme t_ijbe

    Hovoo += contract("me,ebij->mbij", f[o, v], t2)

    # 0.5 * tau_ijef <mb||ef>
    Hovoo += contract("efij,mbef->mbij", t2, u[o, v, v, v])

    # P(ij) t_jnbe <mn||ie>

    Hovoo -= contract("ebin,mnej->mbij", t2, u[o, o, v, o])
    Hovoo -= contract("ebjn,mnie->mbij", t2, u[o, o, o, v])
    Hovoo += contract("bejn,mnie->mbij", t2, Looov)

    return Hovoo
