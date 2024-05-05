import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJMixIn')

class NLJMixIn():
    def count_v_terms(self):
        return 0

    def count_u_terms(self):
        return 0

    def count_m_terms(self):
        return 0

    def count_n_terms(self):
        return 0

    def get_jv_names(self):
        return []

    def get_ju_names(self):
        return []

    def get_jm_names(self):
        return []

    def get_jn_names(self):
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
    def need_pa(self):
        return False

    def get_jt_pe_pa_idx(self):
        root = self.get_root_phys()
        i_jt = -1
        i_pe = -1
        i_pa = -1
        for i in range(4):
            flag = root.check_kfes(i)
            if flag == 20:
                i_jt = i
            elif flag == 22:
                i_pe = i
            elif flag == 23:
                i_pa = i
        return i_jt, i_pe, i_pa
 
from petram.phys.phys_model import Phys, PhysModule   
from petram.model import Domain, Bdry, Point, Pair

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
    pass


class NLJ_BasePoint(Point, Phys, NLJMixIn):
    def __init__(self, **kwargs):
        Point.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)
        NLJMixIn.__init__(self)
    pass


class NLJ_BasePair(Pair, Phys, NLJMixIn):
    def __init__(self, **kwargs):
        Pair.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)
        NLJMixIn.__init__(self)

    
class NLJPhysMixIn():
    '''
    MixIn class (contains routines which does not overwrite baseclass routine)
    '''
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
    def nmterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_m_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_m_terms")])

    @property
    def nnterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_n_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_n_terms")])
    
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
        values.append(self.nmterms)
        values.append(self.nnterms)
        return values
    
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
        jmname = self.get_root_phys().extra_vars_basem
        jnname = self.get_root_phys().extra_vars_basen

        if dep_var.startswith(juname):
            return 18
        elif dep_var.startswith(jvname):
            return 19
        elif dep_var.startswith(jmname):
            return 24
        elif dep_var.startswith(jnname):
            return 25

        else:
            assert False, "Should not come here (unknown FES type : " + dep_var

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
    def extra_vars_basem(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "m"
        return basename

    @property
    def extra_vars_basen(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "n"
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

class NLJJhotBase(NLJ_BaseDomain):
    @property
    def need_pe(self):
        return True

    @property
    def need_pa(self):
        if not hasattr(self, 'use_eta'):
            self.use_eta = False
        if not hasattr(self, 'use_xi'):
            self.use_xi = False
        if not hasattr(self, 'use_pi'):
            self.use_pi = False
        return self.use_eta or self.use_xi or self.use_pi

    def get_ju_names(self):
        names = self.current_names_xyz()
        return names[0]

    def get_jv_names(self):
        names = self.current_names_xyz()
        return names[1]

    def get_jm_names(self):
        names = self.current_names_xyz()
        return names[2]

    def get_jn_names(self):
        names = self.current_names_xyz()
        return names[3]

    def count_u_terms(self):
        return len(self.get_ju_names())

    def count_v_terms(self):
        return len(self.get_jv_names())

    def count_m_terms(self):
        return len(self.get_jm_names())

    def count_n_terms(self):
        return len(self.get_jn_names())

    def current_names_xyz(self):
        # all possible names without considering run-condition
        baseu = self.get_root_phys().extra_vars_baseu
        basev = self.get_root_phys().extra_vars_basev
        udiag = [baseu + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]
        vdiag = [basev + self.name() + str(i+1)
                 for i in range(self._count_perp_terms())]

        if self.use_eta or self.use_xi or self.use_pi:
            basem = self.get_root_phys().extra_vars_basem
            basen = self.get_root_phys().extra_vars_basen
            mdiag = [basem + self.name() + str(i+1)
                     for i in range(self._count_perp_terms())]
            ndiag = [basen + self.name() + str(i+1)
                     for i in range(self._count_perp_terms())]

        else:
            mdiag = []
            ndiag = []

        return udiag, vdiag, mdiag, ndiag

    def get_dep_var_idx(self, dep_var):
        names = self.current_names_xyz()
        udiag, vdiag, mdiag, ndiag = names
        if dep_var in udiag:
            idx = udiag.index(dep_var)
            umode = True
            flag = 18
        elif dep_var in vdiag:
            idx = vdiag.index(dep_var)
            umode = False
            flag = 19
        elif dep_var in mdiag:
            idx = mdiag.index(dep_var)
            umode = True
            flag = 24
        elif dep_var in ndiag:
            idx = ndiag.index(dep_var)
            umode = False
            flag = 25
        else:
            assert False, "should not come here" + str(dep_var)
        return idx, umode, flag
    
    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)

        dep_var = root.kfes2depvar(kfes)

        names = self.current_names_xyz()
        udiag, vdiag, mdiag, ndiag = names
        all_names = udiag + vdiag + mdiag + ndiag

        if dep_var not in all_names:
            return False

        if check == 18:     # u-component
            return True
        elif check == 19:   # v-component
            return True
        elif check == 24:   # m-component
            return True
        elif check == 25:   # n-component
            return True
        else:
            return False

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        from petram.helper.pybilininteg import (PyVectorMassIntegrator,
                                                PyVectorWeakPartialPartialIntegrator,)

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        idx, umode, _flag = self.get_dep_var_idx(dep_var)

        # ju[0], jv[0]    -- constant contribution
        # ju[1:], jv[1:] --- diffusion contribution

        # _B, _dens, _temp, _mass, _charge, _tene, ky, kz = self.vt.make_value_or_expression(
        #    self)

        if idx != 0:
            message = "Add diffusion + mass integrator contribution"
            mat = self._jitted_coeffs["weak_lap_perp"]
            self.add_integrator(engine, 'diffusion', mat, a.AddDomainIntegrator,
                                PyVectorWeakPartialPartialIntegrator,
                                itg_params=(3, 3, (0, -1, -1)))

            if umode:
                dterm = self._jitted_coeffs["dterms"][idx-1]
            else:
                dterm = self._jitted_coeffs["dterms"][idx-1].conj()

            dterm = self._jitted_coeffs["eye3x3"]*dterm
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=(3, 3, ))

        else:  # constant term contribution
            message = "Add mass integrator contribution"
            dterm = self._jitted_coeffs["eye3x3"]*self._jitted_coeffs["dd0"]
            self.add_integrator(engine, 'mass', dterm, a.AddDomainIntegrator,
                                PyVectorMassIntegrator,
                                itg_params=(3, 3, ))
        if real:
            dprint1(message, "(real)", dep_var, idx)
        else:
            dprint1(message, "(imag)", dep_var, idx)



