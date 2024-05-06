'''

   NLJ: non-local current. mix-in classes

   physics module to handle non-local current

   hot contribution is handled by H1^3

   flag used in this module
    18: Jpu  (J-vector part u-contribution)
    19: Jpv  (J-vector part v-contribution)
    20: Jt   (J-vector total)
    21: Ev  (vector E)
    22: Evpe (vector E perp)
    23: Evpa (vector E para)

'''
from petram.model import Domain, Bdry, Point, Pair
from petram.phys.phys_model import Phys, PhysModule
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ_MixInl')


class NLJMixIn():
    def count_v_terms(self):
        return 0

    def count_u_terms(self):
        return 0

    def get_jv_names(self):
        return []

    def get_ju_names(self):
        return []

    @property
    def use_e(self):
        return self.get_root_phys().use_e

    @property
    def use_pe(self):
        return self.get_root_phys().use_pe

    @property
    def use_pa(self):
        return self.get_root_phys().use_pa

    @property
    def need_e(self):
        return False

    @property
    def need_pe(self):
        return False

    @property
    def use_pa(self):
        return False

    def get_jt_e_pe_pa_idx(self):
        root = self.get_root_phys()
        i_jt = -1
        i_pe = -1
        i_pa = -1
        for i in range(4):
            flag = root.check_kfes(i)
            if flag == 20:
                i_jt = i
            elif flag == 21:
                i_e = i
            elif flag == 22:
                i_pe = i
            elif flag == 23:
                i_pa = i
        return i_jt, i_e, i_pe, i_pa

    def get_dep_var_idx(self, dep_var):
        names = self.current_names_xyz()
        udiag, vdiag = names
        if dep_var in udiag:
            idx = udiag.index(dep_var)
            umode = True
            flag = 18
        elif dep_var in vdiag:
            idx = vdiag.index(dep_var)
            umode = False
            flag = 19
        else:
            assert False, "should not come here" + str(dep_var)
        return idx, umode, flag


class NLJ_BaseDomain(Domain, Phys, NLJMixIn):
    def __init__(self, **kwargs):
        Domain.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)
        NLJMixIn.__init__(self)


class NLJ_BaseBdry(Bdry, Phys, NLJMixIn):
    def __init__(self, **kwargs):
        Bdry.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)
        NLJMixIn.__init__(self)


class NLJ_BasePoint(Point, Phys, NLJMixIn):
    def __init__(self, **kwargs):
        Point.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)
        NLJMixIn.__init__(self)


class NLJ_BasePair(Pair, Phys, NLJMixIn):
    def __init__(self, **kwargs):
        Pair.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)
        NLJMixIn.__init__(self)


class NLJ_PhysModule(PhysModule):
    @property
    def nuterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_u_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_u_terms")])

    @property
    def nvterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_v_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_v_terms")])

    @property
    def nterms(self):
        '''
        number of terms
        '''
        values = [0]
        if self.use_e:
            values.append(1)
        if self.use_pe:
            values.append(1)
        if self.use_pa:
            values.append(1)
        values.append(1)
        values.append(self.nuterms)
        values.append(self.nvterms)
        return values

    @property
    def extra_vars_basev(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "v"
        return basename

    @property
    def extra_vars_baseu(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "u"
        return basename

    @property
    def use_pe(self):
        for x in self.walk_enabled():
            if hasattr(x, "need_pe"):
                if x.need_pe:
                    return True
        return False

    @property
    def use_pa(self):
        for x in self.walk_enabled():
            if hasattr(x, "need_pa"):
                if x.need_pa:
                    return True
        return False

    @property
    def use_e(self):
        for x in self.walk_enabled():
            if hasattr(x, "need_e"):
                if x.need_e:
                    return True
        return False

    def kfes2depvar(self, kfes):
        root = self.get_root_phys()
        dep_vars = root.dep_vars
        dep_var = dep_vars[kfes]
        return dep_var

    def check_kfes(self, kfes):
        values = []
        if self.use_e:
            values.append(21)
        if self.use_pe:
            values.append(22)
        if self.use_pa:
            values.append(23)
        values.append(20)

        if kfes < len(values):
            return values[kfes]

        dep_var = self.kfes2depvar(kfes)
        dep_vars = self.dep_vars
        jvname = self.get_root_phys().extra_vars_basev
        juname = self.get_root_phys().extra_vars_baseu

        if dep_var.startswith(juname):
            return 18
        elif dep_var.startswith(jvname):
            return 19
        else:
            assert False, "Should not come here (unknown FES type : " + dep_var


class NLJ_Jhot(NLJ_BaseDomain):

    @property
    def need_e(self):
        return True

    def get_itg_param(self):
        raise NotImplementedError(
            "get_itg_params must be implemented in subclass")

    def get_ju_names(self):
        names = self.current_names_xyz()
        return names[0]

    def get_jv_names(self):
        names = self.current_names_xyz()
        return names[1]

    def count_u_terms(self):
        return len(self.get_ju_names())

    def count_v_terms(self):
        return len(self.get_jv_names())

    def current_names_xyz(self):
        # all possible names without considering run-condition
        baseu = self.get_root_phys().extra_vars_baseu
        basev = self.get_root_phys().extra_vars_basev
        udiag = [baseu + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]
        vdiag = [basev + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]

        return udiag, vdiag

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)

        dep_var = root.kfes2depvar(kfes)

        names = self.current_names_xyz()
        udiag, vdiag = names
        all_names = udiag + vdiag

        if dep_var not in all_names:
            return False

        if check == 18:     # u-component
            return True
        elif check == 19:   # v-component
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        root = self.get_root_phys()
        dep_vars = root.dep_vars

        names = self.current_names_xyz()
        udiag, vdiag = names

        root = self.get_root_phys()
        i_jt, i_e, _i_pe, _i_pa = self.get_jt_e_pe_pa_idx()
        assert i_jt >= 0 and i_e >= 0, "Jt or E is not found in dependent variables."

        loc = []
        for name in udiag + vdiag:
            loc.append((name, dep_vars[i_e], 1, 1))
            loc.append((dep_vars[i_jt], name, 1, 1))
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        from petram.helper.pybilininteg import (PyVectorMassIntegrator,
                                                PyVectorWeakPartialPartialIntegrator,)

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        idx, umode, flag = self.get_dep_var_idx(dep_var)

        # ju[0], jv[0]    -- constant contribution
        # ju[1:], jv[1:] --- diffusion contribution

        itg2, itg3 = self.get_itg_params()

        if idx != 0:
            message = "Add diffusion + mass integrator contribution"
            mat = self._jitted_coeffs["weak_lap_perp"]
            self.add_integrator(engine, 'diffusion', mat, a.AddDomainIntegrator,
                                PyVectorWeakPartialPartialIntegrator,
                                itg_params=itg3)

            if umode:
                dterm = self._jitted_coeffs["dterms"][idx-1]
            else:
                dterm = self._jitted_coeffs["dterms"][idx-1].conj()

            dterm = self._jitted_coeffs["eye3x3"]*dterm
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=itg2)

        else:  # constant term contribution
            message = "Add mass integrator contribution"
            dterm = self._jitted_coeffs["eye3x3"]*self._jitted_coeffs["dd0"]
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=itg2)
        if real:
            dprint1(message, "(real)", dep_var, idx)
        else:
            dprint1(message, "(imag)", dep_var, idx)

    def add_mix_contribution2(self, engine, mbf, row, col, is_trans, _is_conj,
                              real=True):
        '''
        fill mixed contribution
        '''
        from petram.helper.pybilininteg import (PyVectorMassIntegrator,
                                                PyVectorPartialIntegrator,
                                                PyVectorPartialPartialIntegrator)

        root = self.get_root_phys()
        dep_vars = root.dep_vars

        meye = self._jitted_coeffs["meye3x3"]
        mbcross = self._jitted_coeffs["mbcross"]
        mbcrosst = self._jitted_coeffs["mbcrosst"]
        jomega = self._jitted_coeffs["jomega"]
        mbperp = self._jitted_coeffs["mbperp"]

        if real:
            dprint1("Add mixed cterm contribution(real)"  "r/c",
                    row, col, is_trans)
        else:
            dprint1("Add mixed cterm contribution(imag)"  "r/c",
                    row, col, is_trans)

        i_jt, i_e, _i_pe, _i_pa = self.get_jt_e_pe_pa_idx()
        itg2, itg3 = self.get_itg_params()

        if col == dep_vars[i_e]:   # E -> Ju, Jv
            idx, umode, flag = self.get_dep_var_idx(row)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if umode:
                if self.use_sigma:
                    ccoeff = mbperp*slot["diag+diagi"]
                    self.add_integrator(engine,
                                        'mass',
                                        ccoeff,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)

                if self.use_delta:
                    ccoeff = mbcross*slot["xy+xyi"]
                    self.add_integrator(engine,
                                        'mass',
                                        ccoeff,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)

                if self.use_tau:
                    mat2 = self._jitted_coeffs["mtau_rank2"]*slot["cl+cli"]
                    mat3 = self._jitted_coeffs["mtau_rank3"]*slot["cl+cli"]
                    mat4 = self._jitted_coeffs["mtau_rank4"]*slot["cl+cli"]

                    self.add_integrator(engine,
                                        'mat2',
                                        mat2,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)

                    self.add_integrator(engine,
                                        'mat3',
                                        mat3,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=itg3)

                    self.add_integrator(engine,
                                        'mat4',
                                        mat4,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialPartialIntegrator,
                                        itg_params=itg3)

                if self.use_eta:
                    mat2 = self._jitted_coeffs["meta_rank2"] * \
                        slot["eta+etai"]
                    mat3 = self._jitted_coeffs["meta_rank3"] * \
                        slot["eta+etai"]

                    self.add_integrator(engine,
                                        'mat2',
                                        mat2,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)

                    self.add_integrator(engine,
                                        'mat3',
                                        mat3,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=itg3)

                if self.use_xi:
                    mat3 = self._jitted_coeffs["mxi_rank3"] * \
                        slot["xi+xii"]

                    self.add_integrator(engine,
                                        'mat3',
                                        mat3,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=itg3)

                #ccoeff = slot["(diag1+diagi1)*Mpara"]
                # self.fill_divgrad_matrix(
                #    engine, mbf, rowi, colj, ccoeff, real, kz=kz)
            else:
                # equivalent to -1j*omega (use 1j*omega since diagnoal is one)
                ccoeff = jomega.conj()
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=itg2)

            return
        if row == dep_vars[i_jt]:  # Ju, Jv -> Jt
            idx, umode, flag = self.get_dep_var_idx(col)

            if idx == 0:
                slot = self._jitted_coeffs["c0"]
            else:
                slot = self._jitted_coeffs["cterms"][idx-1]

            if umode:
                # equivalent to -1j*omega (use 1j*omega since diagnoal is one)
                ccoeff = jomega
                self.add_integrator(engine,
                                    'mass',
                                    ccoeff,
                                    mbf.AddDomainIntegrator,
                                    PyVectorMassIntegrator,
                                    itg_params=itg2)

            else:
                if self.use_sigma:
                    ccoeff = mbperp*slot["conj(diag-diagi)"]
                    self.add_integrator(engine, 'mass',
                                        ccoeff,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)

                if self.use_delta:
                    ccoeff = mbcrosst*slot["conj(xy-xyi)"]
                    self.add_integrator(engine,
                                        'mass',
                                        ccoeff,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)

                if self.use_tau:
                    mat2 = self._jitted_coeffs["mtau_rank2t"] * \
                        slot["conj(cl-cli)"]
                    mat3 = self._jitted_coeffs["mtau_rank3t"] * \
                        slot["conj(cl-cli)"]
                    mat4 = self._jitted_coeffs["mtau_rank4t"] * \
                        slot["conj(cl-cli)"]

                    self.add_integrator(engine,
                                        'mat2',
                                        mat2,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)
                    self.add_integrator(engine,
                                        'mat3',
                                        mat3,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=itg3)
                    self.add_integrator(engine,
                                        'mat4',
                                        mat4,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialPartialIntegrator,
                                        itg_params=itg3)

                if self.use_eta:
                    mat2 = self._jitted_coeffs["meta_rank2t"] * \
                        slot["conj(eta-etai)"]
                    mat3 = self._jitted_coeffs["meta_rank3t"] * \
                        slot["conj(eta-etai)"]

                    self.add_integrator(engine,
                                        'mat2',
                                        mat2,
                                        mbf.AddDomainIntegrator,
                                        PyVectorMassIntegrator,
                                        itg_params=itg2)
                    self.add_integrator(engine,
                                        'mat3',
                                        mat3,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=itg3)
                if self.use_xi:
                    mat3 = self._jitted_coeffs["mxi_rank3t"] * \
                        slot["conj(xi-xii)"]

                    self.add_integrator(engine,
                                        'mat3',
                                        mat3,
                                        mbf.AddDomainIntegrator,
                                        PyVectorPartialIntegrator,
                                        itg_params=itg3)

              #ccoeff = slot["conj(diag1-diagi1)*Mpara"]
              # self.fill_divgrad_matrix(
              #    engine, mbf, rowi, colj, ccoeff, real, kz=kz)
            return

        dprint1("No mixed-contribution"  "r/c", row, col, is_trans)
