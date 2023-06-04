'''

   RFsheath3D module

   physics module to RF sheath in 3D

'''
import sys
import numpy as np

from petram.model import Domain, Bdry, Point, Pair
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('RFSheath3D_Model')

model_basename = 'RFsheath3D'

try:
    import rfsheath_subroutines
except ImportError:
    import petram.mfem_model as mm
    if mm.has_addon_access not in ["any", "rfsheath"]:
        sys.modules[__name__].dependency_invalid = True


class RFsheath3D_DefDomain(Domain, Phys):
    can_delete = False

    def __init__(self, **kwargs):
        super(RFsheath3D_DefDomain, self).__init__(**kwargs)

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass


class RFsheath3D_DefBdry(Bdry, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(RFsheath3D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(RFsheath3D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class RFsheath3D_DefPoint(Point, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(RFsheath3D_DefPoint, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(RFsheath3D_DefPoint, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['']
        return v


class RFsheath3D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = False

    def __init__(self, **kwargs):
        super(RFsheath3D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(RFsheath3D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v


class RFsheath3D(PhysModule):
    dim_fixed = True

    def __init__(self, **kwargs):
        super(RFsheath3D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = RFsheath3D_DefDomain()
        self['Boundary'] = RFsheath3D_DefBdry()

    @property
    def dep_vars(self):
        ret = self.dep_vars_base
        ret = [x + self.dep_vars_suffix for x in ret]
        return ret

    @property
    def dep_vars_base(self):
        val = self.dep_vars_base_txt.split(',')
        return val

    @property
    def dep_vars0(self):
        val = self.dep_vars_base_txt.split(',')
        return [x + self.dep_vars_suffix for x in val]

    @property
    def der_vars(self):
        return []

    @property
    def vdim(self):
        return [1, 1]

    @vdim.setter
    def vdim(self, val):
        pass

    def fes_order(self, idx):
        self.vt_order.preprocess_params(self)
        if idx == 0:
            return self.order
        return self.order-1

    def get_fec_type(self, idx):
        '''
        H1 
        H1v2 (vector dim)
        ND
        RT
        '''
        values = ['H1', 'L2']
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        return [(v[0], 'H1_FECollection'),
                (v[1], 'L2_FECollection'), ]

    def postprocess_after_add(self, engine):
        try:
            sdim = engine.meshes[0].SpaceDimension()
        except:
            return
        if sdim == 3:
            self.ind_vars = 'x, y, z'
            self.ndim = 2
        else:
            import warnings
            warnings.warn("Geometry for RFsheath3D must be 3D")
            return

    def is_complex(self):
        return self.is_complex_valued

    def attribute_set(self, v):
        v = super(RFsheath3D, self).attribute_set(v)
        v["element"] = 'H1_FECollection, L2_FECollection'
        v["ndim"] = 2
        v["ind_vars"] = 'x, y, z'
        v["dep_vars_suffix"] = ''
        v["dep_vars_base_txt"] = 'Vsh, Fmg'
        v["is_complex_valued"] = True
        v["paired_var"] = None

        return v

    def panel1_param(self):
        from petram.utils import pv_panel_param

        panels = super(RFsheath3D, self).panel1_param()

        a, b = self.get_var_suffix_var_name_panel()
        c = pv_panel_param(self, "EM3D1 model")

        panels.extend([
            ["independent vars.", self.ind_vars, 0, {}],
            a, b,
            ["derived vars.", ','.join(self.der_vars), 2, {}],
            c])

        return panels

    def get_panel1_value(self):
        names = self.dep_vars_base_txt
        names2 = ', '.join(self.der_vars)
        val = super(RFsheath3D, self).get_panel1_value()

        from petram.utils import pv_get_gui_value
        gui_value, self.paired_var = pv_get_gui_value(self, self.paired_var)

        val.extend([self.ind_vars,
                    self.dep_vars_suffix,
                    names, names2, gui_value, ])
        return val

    def import_panel1_value(self, v):
        import ifigure.widgets.dialog as dialog

        v = super(RFsheath3D, self).import_panel1_value(v)
        self.ind_vars = str(v[0])
        self.is_complex_valued = False
        self.dep_vars_suffix = str(v[1])
        self.element = 'H1_FECollection, L2_FECollection'
        self.dep_vars_base_txt = ', '.join(
            [x.strip() for x in str(v[2]).split(',')])

        from petram.utils import pv_from_gui_value
        self.paired_var = pv_from_gui_value(self, v[4])

        return True

    def get_possible_domain(self):
        from petram.phys.rfsheath3d.asymptotic import RFsheath3D_Asymptotic
        doms = [RFsheath3D_Asymptotic]
        doms.extend(super(RFsheath3D, self).get_possible_domain())
        return doms

    def get_possible_bdry(self):
        from petram.phys.rfsheath3d.essential import RFsheath3D_Essential
        from petram.phys.rfsheath3d.natural import RFsheath3D_Natural

        bdrs = [RFsheath3D_Essential, RFsheath3D_Natural]
        bdrs.extend(super(RFsheath3D, self).get_possible_bdry())
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

        if name == dep_vars[0]:
            add_scalar(v, name, "", ind_vars, solr, soli)

        elif name == dep_vars[1]:
            for k, suffix in enumerate(ind_vars):
                nn = name + suffix
                v[nn] = GFScalarVariable(solr, soli, comp=k+1)
        else:
            pass
        return v
