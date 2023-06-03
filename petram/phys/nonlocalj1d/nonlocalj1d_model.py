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
    import nonlocalj_subroutines
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
    dim_fixed = False
    def __init__(self, **kwargs):
        super(NonlocalJ1D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = NonlocalJ1D_DefDomain()
        self['Boundary'] = NonlocalJ1D_DefBdry()

    @property
    def dep_vars(self):
        ret = ["phi"+self.dep_vars_suffix,
               "V"+self.dep_vars_suffix]
        return ret

    @property
    def dep_vars_base(self):
        val = self.dep_vars_base_txt.split(',')[0]
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

    def get_fec_type(self, idx):
        '''
        H1 
        H1v2 (vector dim)
        ND
        RT
        '''
        values = ['L2', 'H1']
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        return [(v[0], 'L2_FECollection'),
                (v[1], 'H1_FECollection'), ]

    def postprocess_after_add(self, engine):
        try:
            sdim = engine.meshes[0].SpaceDimension()
        except:
            return
        if sdim == 3:
            self.ind_vars = 'x, y, z'
            self.ndim = 3
        elif sdim == 2:
            self.ind_vars = 'x, y'
            self.ndim = 2
        elif sdim == 1:
            self.ind_vars = 'x'
            self.ndim = 1
        else:
            pass

    def is_complex(self):
        return self.is_complex_valued

    def attribute_set(self, v):
        v = super(NonlocalJ1D, self).attribute_set(v)
        v["element"] = 'L2_FECollection, H1_FECollection'
        v["dim"] = 2
        v["ind_vars"] = 'x, y, z'
        v["dep_vars_suffix"] = ''
        v["dep_vars_base_txt"] = 'Vsh, Fmag'
        v["is_complex_valued"] = False
        v["generate_grad_fespace"] = False
        return v

    def panel1_param(self):
        sdim = self.geom_dim

        import wx
        panels = super(NonlocalJ1D, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        panels.extend([
            ["independent vars.", self.ind_vars, 0, {}],
            a, b,
            ["derived vars.", ','.join(self.der_vars), 2, {}],
            ["predefined ns vars.", txt_predefined, 2, {}],
            ["generate vector", self.generate_grad_fespace, 3, {"text": ' '}], ])
        return panels

    def get_panel1_value(self):
        names = self.dep_vars_base_txt
        names2 = ', '.join(self.der_vars)
        val = super(NonlocalJ1D, self).get_panel1_value()

        val.extend([self.ind_vars,
                    self.dep_vars_suffix,
                    names, names2, txt_predefined,
                    self.generate_grad_fespace])
        return val

    def import_panel1_value(self, v):
        import ifigure.widgets.dialog as dialog

        v = super(NonlocalJ1D, self).import_panel1_value(v)
        self.ind_vars = str(v[0])
        self.is_complex_valued = False
        self.dep_vars_suffix = str(v[1])
        self.element = "H1_FECollection"
        self.dep_vars_base_txt = ','.join(
            [x.strip() for x in str(v[2]).split(',')])
        self.generate_grad_fespace = bool(v[5])
        return True

    '''
    def get_possible_domain(self):
        from petram.phys.rfsheath3d.wall import NonlocalJ1D_Asymtptic
        doms = super(NonlocalJ1D, self).get_possible_domain()
        doms.extend(NonlocalJ1D_Asymtptic)

        return doms
    '''
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
