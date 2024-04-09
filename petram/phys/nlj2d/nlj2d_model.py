'''

   NLJ: non-local current

   physics module to handle non-local current

'''
from petram.mfem_config import use_parallel
import sys
import numpy as np

from petram.model import Domain, Bdry, Point, Pair
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.vtable import VtableElement, Vtable

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLJ2D_Model')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

txt_predefined = ''
model_basename = 'NLJ2D'


try:
    import petram.phys.nonlocalj2d.nonlocal2d_subs_xxyy
except:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "nonlocalj"]:
        sys.modules[__name__].dependency_invalid = True


class NLJ2D_BaseDomain(Domain, Phys):
    def __init__(self, **kwargs):
        Domain.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)

    def count_x_terms(self):
        return 0

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0

    def get_jx_names(self):
        return []

    def get_jy_names(self):
        return []

    def get_jz_names(self):
        return []


class NLJ2D_DefDomain(NLJ2D_BaseDomain):
    data = (('label1', VtableElement(None,
                                     guilabel=None,
                                     default="Default domain couples non-local curent model with EM2D",
                                     tip="Defualt domain must be always on")),
            ('B', VtableElement('bext', type='array',
                            guilabel='magnetic field',
                            default="=[0,0,0]",
                            tip="external magnetic field")),)

    can_delete = False
    is_secondary_condition = True
    vt = Vtable(data)

    def attribute_set(self, v):
        super(NLJ2D_DefDomain, self).attribute_set(v)
        v['sel_readonly'] = True
        v['sel_index_txt'] = 'all'
        return v
    
    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        ind_vars = self.get_root_phys().ind_vars

        _none, B = self.vt.make_value_or_expression(self)
        
        from petram.phys.nlj2d.nlj2d_subs import build_coefficients
        self._jitted_coeffs = build_coefficients(ind_vars, B, self._global_ns, self._local_ns,)

    def has_bf_contribution(self, kfes):
        root = self.get_root_phys()
        check = root.check_kfes(kfes)
        if check in [12, 13, 14, 15, 16, 17, 2, 8, 9]:  # Exs, Eys, Ezs, Jtx, Jty, Jtz
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

        Exyname = var_s[0]
        Ezname = var_s[1]

        loc = []
        loc.append((dep_vars[0], Exyname, 1, 1))   # Exs
        loc.append((dep_vars[1], Exyname, 1, 1))   # Eys
        loc.append((dep_vars[2], Ezname, 1, 1))    # Ezs
        loc.append((dep_vars[3], Exyname, 1, 1))   # Exps
        loc.append((dep_vars[4], Exyname, 1, 1))   # Eyps
        loc.append((dep_vars[5], Exyname, 1, 1))   # Eyps
        loc.append((dep_vars[3], Ezname, 1, 1))    # Exps
        loc.append((dep_vars[4], Ezname, 1, 1))    # Eyps
        loc.append((dep_vars[5], Ezname, 1, 1))    # Ezps
        if root.no_J_E:
            return loc
        loc.append((Exyname, dep_vars[6], 1, 1))   # Jtx -> Exy
        loc.append((Exyname, dep_vars[7], 1, 1))   # Jty -> Exy
        loc.append((Ezname, dep_vars[8], 1, 1))    # Jtz -> Ez
        return loc

    def add_bf_contribution(self, engine, a, real=True, kfes=0):

        root = self.get_root_phys()
        dep_var = root.kfes2depvar(kfes)

        message = "smoothed E and total J"

        if real:
            one = mfem.ConstantCoefficient(1.0)
            self.add_integrator(engine, 'mass', one, a.AddDomainIntegrator,
                                mfem.MassIntegrator)
            dprint1(message, "(real)",  dep_var, kfes)
        else:
            pass

    def add_mix_contribution2(self, engine, mbf, r, c, is_trans, _is_conj,
                              real=True):

        root = self.get_root_phys()
        dep_vars = root.dep_vars

        paired_model = root.paired_model
        mfem_physroot = root.parent
        em2d = mfem_physroot[paired_model]

        var_s = em2d.dep_vars
        freq, omega = em2d.get_freq_omega()

        Exyname = var_s[0]
        Ezname = var_s[1]

        b_mat = self._jitted_coeffs["M_perp"]        

        ### E-total
        if c == Exyname and r == dep_vars[0]:   # Exy -> Exs
            if not real:
                return

            one = mfem.VectorConstantCoefficient(mfem.Vector([-1.0, 0.0]))
            self.add_integrator(engine, 'Ex', one,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)

        elif c == Exyname and r == dep_vars[1]:   # Exy -> Eys
            if not real:
                return

            one = mfem.VectorConstantCoefficient(mfem.Vector([0.0, -1.0]))
            self.add_integrator(engine, 'Ey', one,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)
        elif c == Ezname and r == dep_vars[2]:   # Ez -> Ezs
            if not real:
                return

            one = mfem.ConstantCoefficient(-1.0)
            self.add_integrator(engine, 'Ez', one,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)

        ### E_perp
        elif c == Exyname and r == dep_vars[3]:   # Exy -> Eperp_x
            if not real:
                return
            coeff = b_mat[[0, 1], 0]
            self.add_integrator(engine, 'Epx1', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)

        elif c == Ezname and r == dep_vars[3]:   # Exy -> Eperp_x
            if not real:
                return
            coeff = b_mat[2, 0]
            self.add_integrator(engine, 'Epx2', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)

        elif c == Exyname and r == dep_vars[4]:   # Exy -> Eperp_y
            if not real:
                return

            coeff = b_mat[[0, 1], 1]
            self.add_integrator(engine, 'Epy1', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)
        elif c == Ezname and r == dep_vars[4]:   # Ez -> Eperp_y
            if not real:
                return

            coeff = b_mat[2, 1]
            self.add_integrator(engine, 'Epy2', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)

        elif c == Exyname and r == dep_vars[5]:   # Exy -> Eperp_z
            if not real:
                return

            coeff = b_mat[[0, 1], 2]
            self.add_integrator(engine, 'Epz1', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedDotProductIntegrator)

        elif c == Ezname and r == dep_vars[5]:   # Ez -> Eperp_z
            if not real:
                return

            coeff = b_mat[2, 2]
            self.add_integrator(engine, 'Epz2', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)

        ### J-total
        elif c == dep_vars[6] and r == Exyname:  # -j*omega*Jtx -> Exy
            if real:
                return

            coeff = mfem.VectorConstantCoefficient(mfem.Vector([-omega, 0.0]))
            self.add_integrator(engine, 'cterm', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorProductIntegrator)

        elif c == dep_vars[7] and r == Exyname:  # -j*omega*Jty -> Exy
            if real:
                return

            coeff = mfem.VectorConstantCoefficient(mfem.Vector([0.0, -omega]))
            self.add_integrator(engine, 'cterm', coeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedVectorProductIntegrator)

        elif c == dep_vars[8] and r == Ezname:  # -j*omega*Jtz -> Ez
            if real:
                return

            ccoeff = mfem.ConstantCoefficient(-omega)
            self.add_integrator(engine, 'cterm', ccoeff,
                                mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)
        else:
            assert False, "Should not come here: " + r + "/" + c


class NLJ2D_DefBdry(Bdry, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NLJ2D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NLJ2D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class NLJ2D_DefPoint(Point, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NLJ2D_DefPoint, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NLJ2D_DefPoint, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['']
        return v


class NLJ2D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = False

    def __init__(self, **kwargs):
        super(NLJ2D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NLJ2D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v


class NLJ2D(PhysModule):
    dim_fixed = True

    def __init__(self, **kwargs):
        super(NLJ2D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = NLJ2D_DefDomain()
        self['Boundary'] = NLJ2D_DefBdry()

    @property
    def nxterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_x_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_x_terms")])

    @property
    def nyterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_y_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_y_terms")])

    @property
    def nzterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_z_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_z_terms")])

    @property
    def nterms(self):
        '''
        number of H1(Ex, Ey, Ez), H1(Epx, Epy, Epz), H1(Jx, Jy, Jz), H1(x), H1 (y), H1(z)
        '''
        return (6, 3,
                self.nxterms,
                self.nyterms,
                self.nzterms,)

    def verify_setting(self):
        return True, '', ''

    def kfes2depvar(self, kfes):
        root = self.get_root_phys()
        dep_vars = root.dep_vars
        dep_var = dep_vars[kfes]
        return dep_var

    def check_kfes(self, kfes):
        '''
           return
                0: jxy
                2: jz
                3: jxy contribution
                4: jxyp contribution (div Jxy, or curl Jxy)
                5: jz contribution
                6: jx contribution
                7: jy contribution
                8: jx
                9: jy
               10: eperp
               11: epara
               12: Exs (Ex in H1)
               13: Eys (Ey in H1)
               14: Eys (Ez in H1)
               15: Exps (E-perp x in H1)
               16: Eyps (E-perp y in H1)
               17: Eyps (E-perp z in H1)

        '''
        dep_var = self.kfes2depvar(kfes)

        dep_vars = self.dep_vars
        jzname = self.get_root_phys().extra_vars_basez
        jxname = self.get_root_phys().extra_vars_basex
        jyname = self.get_root_phys().extra_vars_basey

        if dep_var == dep_vars[8]:
            return 2
        elif dep_var == dep_vars[6]:
            return 8
        elif dep_var == dep_vars[7]:
            return 9
        elif dep_var == dep_vars[0]:
            return 12
        elif dep_var == dep_vars[1]:
            return 13
        elif dep_var == dep_vars[2]:
            return 14
        elif dep_var == dep_vars[3]:
            return 15
        elif dep_var == dep_vars[4]:
            return 16
        elif dep_var == dep_vars[5]:
            return 17
        elif dep_var.startswith(jzname):
            return 5
        elif dep_var.startswith(jxname):
            return 6
        elif dep_var.startswith(jyname):
            return 7

    @property
    def dep_vars(self):
        basename = (self.dep_vars_base_txt +
                    self.dep_vars_suffix)
        ret = []

        ret.append(basename+"Exs")  # Exs (E-total)
        ret.append(basename+"Eys")
        ret.append(basename+"Ezs")
        ret.append(basename+"Exps")  # Exps (E-perp)
        ret.append(basename+"Eyps")
        ret.append(basename+"Ezps")
        ret.append(basename+"Jtx")   #Jt (J-total)
        ret.append(basename+"Jty")
        ret.append(basename+"Jtz")

        for x in self["Domain"].walk_enabled():
            if x.count_x_terms() > 0:
                ret.extend(x.get_jx_names())

        for x in self["Domain"].walk_enabled():
            if x.count_y_terms() > 0:
                ret.extend(x.get_jy_names())

        for x in self["Domain"].walk_enabled():
            if x.count_z_terms() > 0:
                ret.extend(x.get_jz_names())

        return ret

    @property
    def dep_vars_base(self):
        val = self.dep_vars_base_txt.split(',')
        return val

    @property
    def dep_vars0(self):
        return self.dep_vars

    @property
    def extra_vars_basex(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "x"
        return basename

    @property
    def extra_vars_basey(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "y"
        return basename

    @property
    def extra_vars_basez(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "z"
        return basename

    @property
    def der_vars(self):
        return []

    @property
    def vdim(self):
        nnE, nnJ, nh1x, nh1y, nh1z = self.nterms
        return [1] * (nnE + nnJ + nh1x + nh1y + nh1z)

    @vdim.setter
    def vdim(self, val):
        pass

    def get_fec_type(self, idx):
        '''
        H1 
        H1v2 (vector dim)
        ND
        RT
        '''
        nnE, nnJ, nh1x, nh1y, nh1z = self.nterms
        values = ['H1'] * (nnE, nnJ + nh1x + nh1y + nh1z)
        return values[idx]

    def get_fec(self):
        v = self.dep_vars

        jxname = self.get_root_phys().extra_vars_basex
        jyname = self.get_root_phys().extra_vars_basey
        jzname = self.get_root_phys().extra_vars_basez

        fecs = []

        for vv in v:
            if vv in v[:9]:
                fecs.append((vv, 'H1_FECollection'))

            elif vv.startswith(jxname):
                fecs.append((vv, 'H1_FECollection'))

            elif vv.startswith(jyname):
                fecs.append((vv, 'H1_FECollection'))

            elif vv.startswith(jzname):
                fecs.append((vv, 'H1_FECollection'))

            else:
                assert False, "should not come here"

        return fecs

    def fes_order(self, idx):
        self.vt_order.preprocess_params(self)

        flag = self.check_kfes(idx)
        if flag == 2:  # jz
            return self.order

        elif flag == 5:  # jz  components
            return self.order

        elif flag == 6:  # jx  components
            return self.order

        elif flag == 7:  # jy components
            return self.order

        elif flag == 8:  # jx
            return self.order

        elif flag == 9:  # jy
            return self.order

        elif flag == 12:  # Exs
            return self.order
        elif flag == 13:  # Eys
            return self.order
        elif flag == 14:  # Ezs
            return self.order
        elif flag == 15:  # Exps
            return self.order
        elif flag == 16:  # Eyps
            return self.order
        elif flag == 17:  # Ezps
            return self.order
        else:
            assert False, "unsupported flag: "+str(flag)

    def postprocess_after_add(self, engine):
        try:
            sdim = engine.meshes[0].SpaceDimension()
        except:
            return

        if sdim == 2:
            self.ind_vars = 'x, y'
            self.ndim = 2
        else:
            import warnings
            warnings.warn("Mesh should be 2D mesh for this model")

    def is_complex(self):
        return self.is_complex_valued

    def attribute_set(self, v):
        v = super(NLJ2D, self).attribute_set(v)

        nnE, nnJ, nh1x, nh1y, nh1z = self.nterms

        elements = "H1_FECollection * "+str(nnE + nnJ + nh1x + nh1y + nh1z)

        v["element"] = elements
        v["dim"] = 1
        v["ind_vars"] = 'x, y'
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

        panels = super(NLJ2D, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        b[0] = "dep. vars. base"
        c = pm_panel_param(self, "EM2D1 model")

        panels.extend([
            ["independent vars.", self.ind_vars, 0, {}],
            a, b,
            ["dep. vars.", ','.join(self.dep_vars), 2, {}],
            ["derived vars.", ','.join(self.der_vars), 2, {}],
            ["predefined ns vars.", txt_predefined, 2, {}],
            c,
            ["use E->J only", False, 3, {"text": ' '}],
        ])

        return panels

    def get_panel1_value(self):
        import textwrap

        names = '\n'.join(textwrap.wrap(', '.join(self.dep_vars), width=50))
        name2 = '\n'.join(textwrap.wrap(', '.join(self.der_vars), width=50))

        val = super(NLJ2D, self).get_panel1_value()

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

        v = super(NLJ2D, self).import_panel1_value(v)

        self.ind_vars = str(v[0])
        self.is_complex_valued = True
        self.dep_vars_suffix = str(v[1])

        nnE, nnJ, nh1x, nh1y, nh1z = self.nterms
        self.element = "H1_FECollection * " + \
            str(nnE + nnJ + nh1x + nh1y + nh1z)

        self.dep_vars_base_txt = (str(v[2]).split(','))[0].strip()

        from petram.utils import pm_from_gui_value
        self.paired_model = pm_from_gui_value(self, v[-2])
        self.no_J_E = bool(v[-1])

        return True

    def get_possible_bdry(self):
        if NLJ2D._possible_constraints is None:
            self._set_possible_constraints('nlj2d')
        bdrs = super(NLJ2D, self).get_possible_bdry()
        return NLJ2D._possible_constraints['bdry'] + bdrs

    def get_possible_domain(self):
        if NLJ2D._possible_constraints is None:
            self._set_possible_constraints('nlj2d')

        doms = super(NLJ2D, self).get_possible_domain()
        return NLJ2D._possible_constraints['domain'] + doms

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

        add_scalar(v, name, "", ind_vars, solr, soli)

        return v

        '''
        dep_vars = self.dep_vars
        sdim = self.geom_dim

        # x
        xnames = []
        basename = self.extra_vars_basex
        xnames.append(basename)
        for x in self["Domain"].walk_enabled():
            if x.count_x_terms() > 0:
                xnames.extend(x.get_jx_names())

        if name in xnames:
            add_scalar(v, name, suffix, ind_vars, solr, soli)

        # y
        ynames = []
        basename = self.extra_vars_basey
        ynames.append(basename)
        for x in self["Domain"].walk_enabled():
            if x.count_y_terms() > 0:
                ynames.extend(x.get_jy_names())

        if name in ynames:
            add_scalar(v, name, suffix, ind_vars, solr, soli)

        # z
        znames = []
        basename = self.extra_vars_basez
        znames.append(basename)
        for x in self["Domain"].walk_enabled():
            if x.count_z_terms() > 0:
                znames.extend(x.get_jz_names())

        if name in znames:
            add_scalar(v, name, suffix, ind_vars, solr, soli)

        return v
        '''
