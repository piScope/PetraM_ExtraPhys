'''

   RFsheath3D module

   physics module to RF sheath in 3D

'''
import sys
import numpy as np

from petram.model import Domain, Bdry, Point, Pair
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NonlocalJ2D_Model')

txt_predefined = ''
model_basename = 'NonlocalJ2D'


try:
    import petram.phys.nonlocalj2d.nonlocal2d_subs_xxyy
except:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "nonlocalj"]:
        sys.modules[__name__].dependency_invalid = True


class NonlocalJ2D_BaseDomain(Domain, Phys):
    def __init__(self, **kwargs):
        Domain.__init__(self, **kwargs)
        Phys.__init__(self, **kwargs)

    def count_xy_terms(self):
        return 0

    def count_p_terms(self):
        return 0

    def count_x_terms(self):
        return 0

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0

    def has_eperp(self):
        return 0

    def has_epara(self):
        return 0

    def has_jxy(self):
        return 0

    def has_jp(self):
        return 0

    def has_jx(self):
        return 0

    def has_jy(self):
        return 0

    def has_jz(self):
        return 0

    def get_jxy_names(self):
        return []

    def get_jp_names(self):
        return []

    def get_jx_names(self):
        return []

    def get_jy_names(self):
        return []

    def get_jz_names(self):
        return []

    @property
    def use_h1(self):
        return self.get_root_phys().use_h1

    @property
    def use_nd(self):
        return self.get_root_phys().use_nd

    @property
    def use_rt(self):
        return self.get_root_phys().use_rt


class NonlocalJ2D_DefDomain(NonlocalJ2D_BaseDomain):
    can_delete = False

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass


class NonlocalJ2D_DefBdry(Bdry, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NonlocalJ2D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NonlocalJ2D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class NonlocalJ2D_DefPoint(Point, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(NonlocalJ2D_DefPoint, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NonlocalJ2D_DefPoint, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['']
        return v


class NonlocalJ2D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = False

    def __init__(self, **kwargs):
        super(NonlocalJ2D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(NonlocalJ2D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v


class NonlocalJ2D(PhysModule):
    dim_fixed = True

    def __init__(self, **kwargs):
        super(NonlocalJ2D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = NonlocalJ2D_DefDomain()
        self['Boundary'] = NonlocalJ2D_DefBdry()

    @property
    def nxyterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_xy_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_xy_terms")])

    @property
    def npterms(self):
        if "Domain" not in self:
            return 0
        return np.sum([x.count_p_terms()
                       for x in self["Domain"].walk_enabled()
                       if hasattr(x, "count_p_terms")])

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
    def has_eperp(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_eperp()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_eperp")]))

    @property
    def has_epara(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_epara()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_epara")]))

    @property
    def has_jxy(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_jxy()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_jxy")]))

    @property
    def has_jp(self):
        if "Domain" not in self:
            return 0
        return int(np.any([x.has_jp()
                           for x in self["Domain"].walk_enabled()
                           if hasattr(x, "has_jp")]))

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
    def use_h1(self):
        return self.discretization == "H1"

    @property
    def use_rt(self):
        return self.discretization == "RT"

    @property
    def use_nd(self):
        return self.discretization == "ND"

    @property
    def nterms(self):
        '''
        number of NE/RT(xy), H1(aux), H1(x), H1 (y), H1(z)
        '''
        return (self.nxyterms + self.has_jxy,
                self.npterms,
                self.nxterms + self.has_jx,
                self.nyterms + self.has_jy,
                self.nzterms + self.has_jz,)

    def verify_setting(self):
        if self.use_h1 and self.use_rt:
            flag = False
        elif self.use_h1 and self.use_nd:
            flag = False
        else:
            flag = True
        return flag, "Incosistent FES", "RT/ND and H1 for xy components are not used simultaneously"

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
        '''
        dep_var = self.kfes2depvar(kfes)
        eperpname = self.get_root_phys().extra_vars_eperp
        eparaname = self.get_root_phys().extra_vars_epara
        jxyname = self.get_root_phys().extra_vars_basexy
        jpname = self.get_root_phys().extra_vars_basep
        jzname = self.get_root_phys().extra_vars_basez
        jxname = self.get_root_phys().extra_vars_basex
        jyname = self.get_root_phys().extra_vars_basey

        if dep_var == jxyname:
            return 0
        elif dep_var == jzname:
            return 2
        elif dep_var == jxname:
            return 8
        elif dep_var == jyname:
            return 9
        elif dep_var == eperpname:
            return 10
        elif dep_var == eparaname:
            return 11
        elif dep_var.startswith(jxyname):
            return 3
        elif dep_var.startswith(jpname):
            return 4
        elif dep_var.startswith(jzname):
            return 5
        elif dep_var.startswith(jxname):
            return 6
        elif dep_var.startswith(jyname):
            return 7

    @property
    def dep_vars(self):
        base = self.dep_vars_base_txt
        ret = []

        if self.has_eperp:
            basename = self.extra_vars_eperp
            ret.append(basename)

        if self.has_epara:
            basename = self.extra_vars_epara
            ret.append(basename)

        if self.has_jxy:
            basename = self.extra_vars_basexy
            ret.append(basename)

        for x in self["Domain"].walk_enabled():
            if x.count_xy_terms() > 0:
                ret.extend(x.get_jxy_names())

        for x in self["Domain"].walk_enabled():
            if x.count_p_terms() > 0:
                ret.extend(x.get_jp_names())

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
            ret = [base + self.dep_vars_suffix + "xy"]
        return ret

    @property
    def dep_vars_base(self):
        val = self.dep_vars_base_txt.split(',')
        return val

    @property
    def dep_vars0(self):
        return self.dep_vars

    @property
    def extra_vars_eperp(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "eperp"
        return basename

    @property
    def extra_vars_epara(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "epara"
        return basename

    @property
    def extra_vars_basexy(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "xy"
        return basename

    @property
    def extra_vars_basep(self):
        base = self.dep_vars_base_txt
        basename = base+self.dep_vars_suffix + "p"
        return basename

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
        nnd, nh1v, nh1x, nh1y, nh1z = self.nterms
        return [1] * (nnd + nh1v + nh1x + nh1y + nh1z)

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
        nnd, nh1v, nh1x, nh1y, nh1z = self.nterms
        if self.use_nd:
            values = ['ND'] * nnd + ['H1'] * (nh1v + nh1x + nh1y + nh1z)
        else:
            values = ['RT'] * nnd + ['H1'] * (nh1v + nh1x + nh1y + nh1z)
        return values[idx]

    def get_fec(self):
        v = self.dep_vars

        eperpname = self.get_root_phys().extra_vars_eperp
        eparaname = self.get_root_phys().extra_vars_epara
        jxyname = self.get_root_phys().extra_vars_basexy
        jpname = self.get_root_phys().extra_vars_basep
        jxname = self.get_root_phys().extra_vars_basex
        jyname = self.get_root_phys().extra_vars_basey
        jzname = self.get_root_phys().extra_vars_basez

        fecs = []

        for vv in v:
            if vv.startswith(jxyname):
                if self.use_nd:
                    fecs.append((vv, 'ND_FECollection'))
                if self.use_rt:
                    fecs.append((vv, 'RT_FECollection'))
                continue

            if vv.startswith(jpname):
                fecs.append((vv, 'H1_FECollection'))
                continue

            if vv == eperpname:
                fecs.append((vv, 'ND_FECollection'))

            elif vv == eparaname:
                fecs.append((vv, 'H1_FECollection'))

            elif vv == jxname:
                fecs.append((vv, 'H1_FECollection'))

            elif self.use_h1 and vv.startswith(jxname):
                fecs.append((vv, 'H1_FECollection'))

            elif vv == jyname:
                fecs.append((vv, 'H1_FECollection'))

            elif self.use_h1 and vv.startswith(jyname):
                fecs.append((vv, 'H1_FECollection'))

            elif vv.startswith(jzname):
                fecs.append((vv, 'H1_FECollection'))

            else:
                assert False, "should not come here"

        return fecs

    def fes_order(self, idx):
        self.vt_order.preprocess_params(self)

        flag = self.check_kfes(idx)
        if flag == 0:  # jxy
            return self.order

        elif flag == 2:  # jz
            return self.order

        elif flag == 3:  # jxy components
            return self.order

        elif flag == 4:  # jpname
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

        elif flag == 10:  # e_perp
            return self.order

        elif flag == 11:  # e_para
            return self.order

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
        v = super(NonlocalJ2D, self).attribute_set(v)

        v['discretization'] = 'H1'
        if not hasattr(self, 'discretization'):
            self.discretization = "H1"
        nnd, nh1v, nh1x, nh1y, nh1z = self.nterms
        if self.use_rt:
            elements = "RT_FECollection * " + \
                str(nnd) + ", H1_FECollection * " + \
                str(nh1v + nh1x + nh1y + nh1z)
        elif self.use_nd:
            elements = "ND_FECollection * " + \
                str(nnd) + ", H1_FECollection * " + \
                str(nh1v + nh1x + nh1y + nh1z)
        else:
            elements = "H1_FECollection * "+str(nh1v + nh1x + nh1y + nh1z)

        v["element"] = elements
        v["dim"] = 1
        v["ind_vars"] = 'x, y'
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

        panels = super(NonlocalJ2D, self).panel1_param()
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
            ["Discretization (Jxy)", None, 0, {}],
        ])

        return panels

    def get_panel1_value(self):
        import textwrap

        names = '\n'.join(textwrap.wrap(', '.join(self.dep_vars), width=50))
        name2 = '\n'.join(textwrap.wrap(', '.join(self.der_vars), width=50))

        val = super(NonlocalJ2D, self).get_panel1_value()

        from petram.utils import pm_get_gui_value
        gui_value, self.paired_model = pm_get_gui_value(
            self, self.paired_model)

        val.extend([self.ind_vars,
                    self.dep_vars_suffix,
                    self.dep_vars_base_txt,
                    names, name2, txt_predefined, gui_value,
                    self.discretization, ])

        return val

    def import_panel1_value(self, v):
        import ifigure.widgets.dialog as dialog

        v = super(NonlocalJ2D, self).import_panel1_value(v)

        self.ind_vars = str(v[0])
        self.is_complex_valued = True
        self.dep_vars_suffix = str(v[1])
        self.discretization = v[-1]

        nnd, nh1v, nh1x, nh1y, nh1z = self.nterms
        if self.use_rt:
            self.element = "RT_FECollection * " + \
                str(nnd) + ", H1_FECollection * " + \
                str(nh1v + nh1x + nh1y + nh1z)
        elif self.use_nd:
            self.element = "ND_FECollection * " + \
                str(nnd) + ", H1_FECollection * " + \
                str(nh1v + nh1x + nh1y + nh1z)

        else:
            self.element = "H1_FECollection * " + \
                str(nh1v + nh1x + nh1y + nh1z)

        self.dep_vars_base_txt = (str(v[2]).split(','))[0].strip()

        from petram.utils import pm_from_gui_value
        self.paired_model = pm_from_gui_value(self, v[-2])

        return True

    def get_possible_domain(self):
        from petram.phys.nonlocalj2d.jhot import NonlocalJ2D_Jhot
        from petram.phys.nonlocalj2d.jxxyy import NonlocalJ2D_Jxxyy
        from petram.phys.nonlocalj2d.jxxyy2 import NonlocalJ2D_Jxxyy2
        from petram.phys.nonlocalj2d.jperp3 import NonlocalJ2D_Jperp3
        from petram.phys.nonlocalj2d.jperp4 import NonlocalJ2D_Jperp4

        doms = [NonlocalJ2D_Jhot, NonlocalJ2D_Jxxyy,
                NonlocalJ2D_Jxxyy2, NonlocalJ2D_Jperp3, NonlocalJ2D_Jperp4]
        doms.extend(super(NonlocalJ2D, self).get_possible_domain())

        return doms

    def get_possible_bdry(self):
        from petram.phys.nonlocalj2d.coldedge import NonlocalJ2D_ColdEdge
        from petram.phys.nonlocalj2d.continuity import NonlocalJ2D_Continuity
        bdrs = [NonlocalJ2D_ColdEdge, NonlocalJ2D_Continuity]
        bdrs.extend(super(NonlocalJ2D, self).get_possible_bdry())
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

        dep_vars = self.dep_vars
        sdim = self.geom_dim

        xynames = []
        if self.has_jxy:
            basename = self.extra_vars_basexy
            xynames.append(basename)

        for x in self["Domain"].walk_enabled():
            if x.count_xy_terms() > 0:
                xynames.extend(x.get_jxy_names())

        if name in xynames:
            add_elements(v, name, suffix, ind_vars,
                         solr, soli, elements=[0, 1])

        pnames = []
        for x in self["Domain"].walk_enabled():
            if x.count_p_terms() > 0:
                pnames.extend(x.get_jp_names())

        if name in pnames:
            add_scalar(v, name, suffix, ind_vars, solr, soli)

        znames = []
        if self.has_jz:
            basename = self.extra_vars_basez
            znames.append(basename)
        for x in self["Domain"].walk_enabled():
            if x.count_z_terms() > 0:
                znames.extend(x.get_jz_names())

        if name in znames:
            add_scalar(v, name, suffix, ind_vars, solr, soli)

        return v
