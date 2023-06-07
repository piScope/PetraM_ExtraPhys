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


class NonlocalJ1D_DefDomain(Domain, Phys):
    can_delete = False

    def __init__(self, **kwargs):
        super(NonlocalJ1D_DefDomain, self).__init__(**kwargs)

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass

    def count_x_terms(self):
        return 0

    def count_y_terms(self):
        return 0

    def count_z_terms(self):
        return 0


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
    def nterms(self):
        '''
        number of h1 basis. it can not be zero. so if it is zero, it returns 1.
        '''
        total = 0
        if self.nxterms > 0:
            total += self.nxterms + 1
        if self.nyterms > 0:
            total += self.nyterms + 1
        if self.nzterms > 0:
            total += self.nzterms + 1

        if total == 0:
            total = 1
        return total

    @property
    def dep_vars(self):
        base = self.dep_vars_base_txt
        ret = []
        nterms = self.nxterms
        if nterms > 0:
            ret.append(base+self.dep_vars_suffix + "x")
            for i in range(nterms):
                ret.append(base+self.dep_vars_suffix+"x_"+str(i+1))
        nterms = self.nyterms
        if nterms > 0:
            ret.append(base+self.dep_vars_suffix + "y")
            for i in range(nterms):
                ret.append(base+self.dep_vars_suffix+"y_"+str(i+1))
        nterms = self.nzterms
        if nterms > 0:
            ret.append(base+self.dep_vars_suffix + "z")
            for i in range(nterms):
                ret.append(base+self.dep_vars_suffix+"z_"+str(i+1))

        if len(ret) == 0:
            ret = [base + self.dep_vars_suffix + "x"]
        return ret

    @property
    def dep_vars_base(self):
        val = self.dep_vars_base_txt.split(',')
        return val

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
        names = ', '.join(self.dep_vars)
        name2 = ', '.join(self.der_vars)
        val = super(NonlocalJ1D, self).get_panel1_value()

        from petram.utils import pm_get_gui_value
        gui_value, self.paired_model = pm_get_gui_value(self, self.paired_model)

        val.extend([self.ind_vars,
                    self.dep_vars_suffix,
                    self.dep_vars_base_txt,
                    names, name2, txt_predefined, gui_value])

        return val

    def import_panel1_value(self, v):
        import ifigure.widgets.dialog as dialog

        v = super(NonlocalJ1D, self).import_panel1_value(v)

        self.ind_vars = str(v[0])
        self.is_complex_valued = False
        self.dep_vars_suffix = str(v[1])
        self.element = "H1_FECollection * "+str(self.nterms)
        self.dep_vars_base_txt = (str(v[2]).split(','))[0].strip()

        from petram.utils import pm_from_gui_value
        self.paired_model = pm_from_gui_value(self, v[-1])

        return True

    def get_possible_domain(self):
        from petram.phys.nonlocalj1d.jxx import NonlocalJ1D_Jxx
        doms = [NonlocalJ1D_Jxx]
        doms.extend(super(NonlocalJ1D, self).get_possible_domain())


        return doms

    '''
    def get_possible_bdry(self):
        from petram.phys.rfsheath3d.wall import NonlocalJ1D_Wall

        bdrs = super(NonlocalJ1D, self).get_possible_bdry()
        return [NonlocalJ1D_Wall]
    '''

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

        if name == dep_vars[0]:
            add_scalar(v, name, "", ind_vars, solr, soli)

        elif name == dep_vars[1]:
            for k, suffix in enumerate(ind_vars):
                nn = name + suffix
                v[nn] = GFScalarVariable(solr, soli, comp=k+1)
        else:
            pass
        return v
