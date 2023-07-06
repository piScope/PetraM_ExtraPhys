'''

   RFsheath3D module

   physics module to RF sheath in 3D

'''
import sys
import numpy as np

from petram.model import Domain, Bdry, Point, Pair
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ1D_Model')

txt_predefined = ''
model_basename = 'NonlocalJ1D'


try:
    import petram.phys.nonlocalj1d.nonlocalj1d_subs
except:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "nonlocalj"]:
        sys.modules[__name__].dependency_invalid = True


class NonlocalJ1D_BaseDomain(Domain, Phys):
    def __init__(self, **kwargs):
        Domain.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)

    def count_x_terms(self):
        return 0

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0

    def has_jx(self):
        return 0

    def has_jy(self):
        return 0

    def has_jz(self):
        return 0

    def get_jx_names(self):
        return []

    def get_jy_names(self):
        return []

    def get_jz_names(self):
        return []


class NonlocalJ1D_DefDomain(NonlocalJ1D_BaseDomain):
    can_delete = False

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass


class NonlocalJ1D_DefBdry(Bdry, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NonlocalJ1D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NonlocalJ1D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class NonlocalJ1D_DefPoint(Point, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NonlocalJ1D_DefPoint, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NonlocalJ1D_DefPoint, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['']
        return v


class NonlocalJ1D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = False

    def __init__(self, **kwargs):
        super(NonlocalJ1D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NonlocalJ1D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v


class NonlocalJ1D(PhysModule):
    dim_fixed = True

    def __init__(self, **kwargs):
        super(NonlocalJ1D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = NonlocalJ1D_DefDomain()
        self['Boundary'] = NonlocalJ1D_DefBdry()

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
    def has_jx(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_jx()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_jx")]))

    @property
    def has_jy(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_jy()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_jy")]))

    @property
    def has_jz(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_jz()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_jz")]))

    @property
    def nterms(self):
        '''
        number of h1 basis. it can not be zero. so if it is zero, it returns 1.
        '''
        return (self.nxterms + self.has_jx +
                self.nyterms + self.has_jy +
                self.nzterms + self.has_jz)

    def kfes2depvar(self, kfes):
        root = self.get_root_phys()
        dep_vars = root.dep_vars
        dep_var = dep_vars[kfes]
        return dep_var

    def check_kfes(self, kfes):
        '''
           return
                0: jx
                1: jy
                2: jz
                3: jx contribution
                4: jy contribution
                5: jz contribution
        '''
        dep_var = self.kfes2depvar(kfes)
        jxname = self.get_root_phys().extra_vars_basex
        jyname = self.get_root_phys().extra_vars_basey
        jzname = self.get_root_phys().extra_vars_basez

        if dep_var == jxname:
            return 0
        elif dep_var == jyname:
            return 1
        elif dep_var == jzname:
            return 2
        elif dep_var.startswith(jxname):
            return 3
        elif dep_var.startswith(jyname):
            return 4
        elif dep_var.startswith(jzname):
            return 5

    @property
    def dep_vars(self):
        base = self.dep_vars_base_txt
        ret = []

        if self.has_jx:
            basename = self.extra_vars_basex
            ret.append(basename)

        for x in self["Domain"].walk_enabled():
            if x.count_x_terms() > 0:
                ret.extend(x.get_jx_names())

        if self.has_jy:
            basename = self.extra_vars_basey
            ret.append(basename)

        for x in self["Domain"].walk_enabled():
            if x.count_y_terms() > 0:
                ret.extend(x.get_jy_names())

        if self.has_jz:
            basename = self.extra_vars_basez
            ret.append(basename)

        for x in self["Domain"].walk_enabled():
            if x.count_z_terms() > 0:
                ret.extend(x.get_jz_names())

        if len(ret) == 0:
            ret = [base + self.dep_vars_suffix + "x"]
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
        return [1] * self.nterms

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
        values = ['H1'] * (self.nterms)
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        return [(vv, 'H1_FECollection') for vv in v]

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
            warnings.warn("Mesh should be 1D mesh for this model")

    def is_complex(self):
        return self.is_complex_valued

    def attribute_set(self, v):
        v = super(NonlocalJ1D, self).attribute_set(v)
        v["element"] = "H1_FECollection * "+str(self.nterms)
        v["dim"] = 1
        v["ind_vars"] = 'x'
        v["dep_vars_suffix"] = ''
        v["dep_vars_base_txt"] = 'Jnl'
        v["is_complex_valued"] = True
        v["paired_model"] = None
        return v

    def get_default_ns(self):
        from petram.phys.phys_const import q0, Da, mass_electron
        ns = {'me': mass_electron,
              'Da': Da,
              'q0': q0}
        return ns

    def panel1_param(self):
        from petram.utils import pm_panel_param

        panels = super(NonlocalJ1D, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        b[0] = "dep. vars. base"
        c = pm_panel_param(self, "EM3D1 model")

        panels.extend([
            ["independent vars.", self.ind_vars, 0, {}],
            a, b,
            ["dep. vars.", ','.join(self.dep_vars), 2, {}],
            ["derived vars.", ','.join(self.der_vars), 2, {}],
            ["predefined ns vars.", txt_predefined, 2, {}],
            c, ])

        return panels

    def get_panel1_value(self):
        import textwrap

        names = '\n'.join(textwrap.wrap(', '.join(self.dep_vars), width=50))
        name2 = '\n'.join(textwrap.wrap(', '.join(self.der_vars), width=50))

        val = super(NonlocalJ1D, self).get_panel1_value()

        from petram.utils import pm_get_gui_value
        gui_value, self.paired_model = pm_get_gui_value(
            self, self.paired_model)

        val.extend([self.ind_vars,
                    self.dep_vars_suffix,
                    self.dep_vars_base_txt,
                    names, name2, txt_predefined, gui_value])

        return val

    def import_panel1_value(self, v):
        import ifigure.widgets.dialog as dialog

        v = super(NonlocalJ1D, self).import_panel1_value(v)

        self.ind_vars = str(v[0])
        self.is_complex_valued = True
        self.dep_vars_suffix = str(v[1])
        self.element = "H1_FECollection * "+str(self.nterms)
        self.dep_vars_base_txt = (str(v[2]).split(','))[0].strip()

        from petram.utils import pm_from_gui_value
        self.paired_model = pm_from_gui_value(self, v[-1])

        return True

    def get_possible_domain(self):
        from petram.phys.nonlocalj1d.jxx import NonlocalJ1D_Jxx
        from petram.phys.nonlocalj1d.jperp import NonlocalJ1D_Jperp
        from petram.phys.nonlocalj1d.jperp2 import NonlocalJ1D_Jperp2
        from petram.phys.nonlocalj1d.jperp3 import NonlocalJ1D_Jperp3
        from petram.phys.nonlocalj1d.jhot import NonlocalJ1D_Jhot
        doms = [NonlocalJ1D_Jhot, NonlocalJ1D_Jxx,
                NonlocalJ1D_Jperp, NonlocalJ1D_Jperp2, NonlocalJ1D_Jperp3]
        doms.extend(super(NonlocalJ1D, self).get_possible_domain())

        return doms

    def get_possible_bdry(self):
        from petram.phys.nonlocalj1d.coldedge import NonlocalJ1D_ColdEdge
        bdrs = [NonlocalJ1D_ColdEdge]
        bdrs.extend(super(NonlocalJ1D, self).get_possible_bdry())
        return bdrs

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

        dep_vars = self.dep_vars
        sdim = self.geom_dim

        add_scalar(v, name, "", ind_vars, solr, soli)

        return v
