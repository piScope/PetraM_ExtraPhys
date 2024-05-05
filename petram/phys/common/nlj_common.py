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
    def use_pa(self):
        return False

