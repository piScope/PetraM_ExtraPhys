'''

   NLJ: non-local current

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
from numba import njit
from petram.mfem_config import use_parallel
import sys
import numpy as np

from petram.phys.vtable import VtableElement, Vtable

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ1D_Model')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

txt_predefined = ''
model_basename = 'NLJ1D'

try:
    import petram.phys.nonlocalj2d.nonlocal2d_subs_xxyy
except:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "nonlocalj"]:
        sys.modules[__name__].dependency_invalid = True


from petram.phys.common.nlj_mixins import (NLJ_BaseDomain,
                                           NLJ_BaseBdry,
                                           NLJ_BasePoint,
                                           NLJ_BasePair,
                                           NLJ_PhysModule)
class NLJ1D_DefDomain(NLJ_BaseDomain):
    data = (('label1', VtableElement(None,
                                     guilabel=None,
                                     default="Default domain couples non-local curent model with EM1D",
                                     tip="Defualt domain must be always on")),
            ('B', VtableElement('bext', type='any',
                                guilabel='magnetic field',
                                default="=[0,0,0]",
                                tip="external magnetic field")),)

    can_delete = False
    is_secondary_condition = True
    vt = Vtable(data)

    def attribute_set(self, v):
        super(NLJ1D_DefDomain, self).attribute_set(v)
        v['sel_readonly'] = True
        v['sel_index_txt'] = 'all'
        return v

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):

        _none, bfunc = self.vt.make_value_or_expression(self)

        root = self.get_root_phys()
        mfem_physroot = root.parent
        em1d = mfem_physroot[root.paired_model]
        freq, omega = em1d.get_freq_omega()
        ind_vars = root.ind_vars

        from petram.phys.common.nlj_common import build_common_nlj_coeff

        self._jitted_coeffs = build_common_nlj_coeff(ind_vars, bfunc, omega,
                                                     self._global_ns, self._local_ns,)

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        if check in [20, 21, 22, 23]:
            return True
        return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        root = self.get_root_phys()
        dep_vars = root.dep_vars

        paired_model = root.paired_model
        mfem_physroot = root.parent
        var_s = mfem_physroot[paired_model].dep_vars

        Exname = var_s[0]
        Eyname = var_s[1]
        Ezname = var_s[2]

        l = 0
        if self.use_e:
            l += 1
        if self.use_pe:
            l += 1
        if self.use_pa:
            l += 1

        loc = []
        for i in range(l):
            loc.append((dep_vars[i], Exname, 1, 1))    # Ex -> Ev
            loc.append((dep_vars[i], Eyname, 1, 1))    # Ey -> Ev
            loc.append((dep_vars[i], Ezname, 1, 1))    # Ez -> Ev

        if root.no_J_E:
            return loc
        loc.append((Exname, dep_vars[l], 1, 1))  # Jt -> Ex
        loc.append((Eyname, dep_vars[l], 1, 1))  # Jt -> Ey
        loc.append((Ezname, dep_vars[l], 1, 1))  # Jt -> Ez

        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)
        message = "smoothed E and total J"

        if real:
            one = mfem.VectorConstantCoefficient([1.0, 1.0, 1.0])
            self.add_integrator(engine, 'mass', one, a.AddDomainIntegrator,
                                mfem.VectorMassIntegrator)
            dprint1(message, "(real)", dep_var, kfes)
        else:
            pass

    def add_mix_contribution2(self, engine, mbf, r, c, is_trans, _is_conj,
                              real=True):

        root = self.get_root_phys()
        dep_vars = root.dep_vars

        def get_dep_var_idx(dep_var):
            kfes = dep_vars.index(dep_var)
            return root.check_kfes(kfes)

        paired_model = root.paired_model
        mfem_physroot = root.parent
        em1d = mfem_physroot[paired_model]

        var_s = em1d.dep_vars
        freq, omega = em1d.get_freq_omega()

        Exname = var_s[0]
        Eyname = var_s[1]
        Ezname = var_s[2]

        # E-total
        # 20: Jt   (J-vector total)
        # 21: Ev  (vector E)
        # 22: Evpe (vector E perp)
        # 23: Evpa (vector E para)

        if real:
            dprint1("Add mixed cterm contribution(real)"  "r/c",
                    r, c, is_trans)
        else:
            dprint1("Add mixed cterm contribution(imag)"  "r/c",
                    r, c, is_trans)

        from petram.helper.pybilininteg import PyVectorMassIntegrator
        if c == Exname:  # Ex -> Ev, Evpe, Evpa
            flag = get_dep_var_idx(r)
            if flag not in [21, 22, 23]:
                assert False, "should not come here: " + str(flag)

            if flag == 21:
                coeff = self._jitted_coeffs["mproj_x"]
            elif flag == 22:
                coeff = self._jitted_coeffs["mb_perp_x"]
            else:
                coeff = self._jitted_coeffs["mb_para_x"]
            shape = (3, 1)

        elif c == Eyname:  # Ey -> Ev, Evpe, Evpa
            flag = get_dep_var_idx(r)
            if flag not in [21, 22, 23]:
                assert False, "should not come here: " + str(flag)

            if flag == 21:
                coeff = self._jitted_coeffs["mproj_y"]
            elif flag == 22:
                coeff = self._jitted_coeffs["mb_perp_y"]
            else:
                coeff = self._jitted_coeffs["mb_para_y"]
            shape = (3, 1)

        elif c == Ezname:  # Ez -> Ev, Evpe, Evpa
            flag = get_dep_var_idx(r)
            if flag not in [21, 22, 23]:
                assert False, "should not come here: " + str(flag)

            if flag == 21:
                coeff = self._jitted_coeffs["mproj_z"]
            elif flag == 22:
                coeff = self._jitted_coeffs["mb_perp_z"]
            else:
                coeff = self._jitted_coeffs["mb_para_z"]
            shape = (3, 1)

        elif r == Exname:  # Jty -> Ex
            flag = get_dep_var_idx(c)
            if flag != 20:
                assert False, "should not come here: " + str(flag)

            coeff = self._jitted_coeffs["one_x"]
            shape = (1, 3)

        elif r == Eyname:  # Jty -> Ey
            flag = get_dep_var_idx(c)
            if flag != 20:
                assert False, "should not come here: " + str(flag)

            coeff = self._jitted_coeffs["one_y"]
            shape = (1, 3)

        elif r == Ezname:  # Jty -> Ez
            flag = get_dep_var_idx(c)
            if flag != 20:
                assert False, "should not come here: " + str(flag)

            coeff = self._jitted_coeffs["one_z"]
            shape = (1, 3)

        else:
            assert False, "should not come here"

        self.add_integrator(engine, 'mass',
                            coeff,
                            mbf.AddDomainIntegrator,
                            PyVectorMassIntegrator,
                            itg_params=shape)


class NLJ1D_DefBdry(NLJ_BaseBdry):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NLJ1D_DefBdry, self).__init__(**kwargs)

    def attribute_set(self, v):
        super(NLJ1D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class NLJ1D_DefPoint(NLJ_BasePoint):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NLJ1D_DefPoint, self).__init__(**kwargs)

    def attribute_set(self, v):
        super(NLJ1D_DefPoint, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['']
        return v


class NLJ1D_DefPair(NLJ_BasePair):
    can_delete = False
    is_essential = False
    is_complex = False

    def __init__(self, **kwargs):
        super(NLJ1D_DefPair, self).__init__(**kwargs)

    def attribute_set(self, v):
        super(NLJ1D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v


class NLJ1D(NLJ_PhysModule):
    dim_fixed = True

    def __init__(self, **kwargs):
        super(NLJ1D, self).__init__()

        self['Domain'] = NLJ1D_DefDomain()
        self['Boundary'] = NLJ1D_DefBdry()


    def verify_setting(self):
        return True, '', ''

    @property
    def dep_vars(self):
        basename = (self.dep_vars_base_txt +
                    self.dep_vars_suffix)
        ret = []

        if self.use_e:
            ret.append(basename+"E")    # E vector (H1-3)
        if self.use_pe:
            ret.append(basename+"Epe")  # E pe vector (H1-3)
        if self.use_pa:
            ret.append(basename+"Epa")  # E pa vector (H1-3)
        ret.append(basename+"Jt")   # Jt (J-total) (H1-3)

        for x in self["Domain"].walk_enabled():
            if x.count_v_terms() > 0:
                ret.extend(x.get_jv_names())
            if x.count_u_terms() > 0:
                ret.extend(x.get_ju_names())

        return ret

    @property
    def dep_vars_base(self):
        val = self.dep_vars_base_txt.split(',')
        return val

    @property
    def dep_vars0(self):
        return self.dep_vars

    @property
    def der_vars(self):
        return []

    @property
    def vdim(self):
        return [3] * sum(self.nterms)

    @vdim.setter
    def vdim(self, val):
        pass

    def get_fec_type(self, idx):
        '''
        H1 array
        '''
        values = ['H1'] * sum(self.nterms)
        return values[idx]

    def get_fec(self):
        fecs = [(x, 'H1_FECollection') for x in self.dep_vars]
        return fecs

    def fes_order(self, idx):
        self.vt_order.preprocess_params(self)

        flag = self.check_kfes(idx)
        if flag in [18, 19, 20, 21, 22, 23]:
            return self.order
        assert False, "unsupported flag: "+str(flag)

    def postprocess_after_add(self, engine):
        try:
            sdim = engine.meshes[0].SpaceDimension()
        except:
            return

        if sdim == 1:
            self.ind_vars = 'x'
            self.ndim = 1
        else:
            import warnings
            warnings.warn("Mesh should be 2D mesh for this model")

    def is_complex(self):
        return self.is_complex_valued

    def attribute_set(self, v):
        v = super(NLJ1D, self).attribute_set(v)
        #elements = "H1_FECollection * "+str(sum(self.nterms))
        elements = "H1_FECollection * 2"

        v["element"] = elements
        v["dim"] = 1
        v["ind_vars"] = 'x'
        v["dep_vars_suffix"] = ''
        v["dep_vars_base_txt"] = 'Nlj'
        v["is_complex_valued"] = True
        v["paired_model"] = None
        v["no_J_E"] = False  # option to suppress feedback from nonlocal current to E
        return v

    def get_default_ns(self):
        from petram.phys.phys_const import q0, Da, mass_electron
        ns = {'me': mass_electron,
              'Da': Da,
              'q0': q0}
        return ns

    def panel1_param(self):
        from petram.utils import pm_panel_param

        panels = super(NLJ1D, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        b[0] = "dep. vars. base"
        c = pm_panel_param(self, "EM1D1 model")

        panels.extend([
            ["independent vars.", self.ind_vars, 0, {}],
            a, b,
            ["dep. vars.", ','.join(self.dep_vars), 2, {}],
            ["derived vars.", ','.join(self.der_vars), 2, {}],
            ["predefined ns vars.", txt_predefined, 2, {}],
            c,
            ["suppress J-> E", False, 3, {"text": ' '}],
        ])

        return panels

    def get_panel1_value(self):
        import textwrap

        names = '\n'.join(textwrap.wrap(', '.join(self.dep_vars), width=50))
        name2 = '\n'.join(textwrap.wrap(', '.join(self.der_vars), width=50))

        val = super(NLJ1D, self).get_panel1_value()

        from petram.utils import pm_get_gui_value
        gui_value, self.paired_model = pm_get_gui_value(
            self, self.paired_model)

        val.extend([self.ind_vars,
                    self.dep_vars_suffix,
                    self.dep_vars_base_txt,
                    names, name2, txt_predefined, gui_value,
                    self.no_J_E])

        return val

    def import_panel1_value(self, v):
        import ifigure.widgets.dialog as dialog

        v = super(NLJ1D, self).import_panel1_value(v)

        self.ind_vars = str(v[0])
        self.is_complex_valued = True
        self.dep_vars_suffix = str(v[1])

        self.element = "H1_FECollection * " + \
            str(sum(self.nterms))

        self.dep_vars_base_txt = (str(v[2]).split(','))[0].strip()

        from petram.utils import pm_from_gui_value
        self.paired_model = pm_from_gui_value(self, v[-2])
        self.no_J_E = bool(v[-1])

        return True

    def get_possible_bdry(self):
        if NLJ1D._possible_constraints is None:
            self._set_possible_constraints('nlj1d')
        bdrs = super(NLJ1D, self).get_possible_bdry()
        return NLJ1D._possible_constraints['bdry'] + bdrs

    def get_possible_domain(self):
        if NLJ1D._possible_constraints is None:
            self._set_possible_constraints('nlj1d')

        doms = super(NLJ1D, self).get_possible_domain()
        return NLJ1D._possible_constraints['domain'] + doms

    def get_possible_point(self):
        '''
        To Do. Support point source
        '''
        return []

    def get_possible_pair(self):
        return []

    def add_variables(self, v, name, solr, soli=None):
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_elements
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant
        from petram.helper.variables import GFScalarVariable

        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        suffix = self.dep_vars_suffix

        #from petram.helper.variables import TestVariable
        #v['debug_test'] =  TestVariable()

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)

        add_components(v, name, "", ind_vars+["y", "z"], solr, soli)

        return v
