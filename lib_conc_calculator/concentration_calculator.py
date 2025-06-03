import numpy as np
from units_constants import *
from utilites import *  

class PlasmaValuesCalculator:
    def __init__(self, plasma_parameters, k_B=1.38e-23, eV=1.602e-19):

        self.lambda_m = plasma_parameters.get("lambda_m")
        self.mu_Cu = plasma_parameters.get("mu_Cu")
        self.f_ik = plasma_parameters.get("f_ik")
        self.g_i = plasma_parameters.get("g_i")
        self.E_i_eV = plasma_parameters.get("E_i")
        self.E_k_eV = plasma_parameters.get("E_k")
        self.k_B = k_B


    def delta_lambda_doppler_m(self,t_K):
        '''
        Return: d_lambda_d
        '''
        lambda_cm = self.lambda_m  * 1e2
        d_lambda_d_cm = 7.16e-7 * lambda_cm * np.sqrt(t_K / self.mu_Cu)
        d_lambda_dopler_m = d_lambda_d_cm * 1e-2
        return d_lambda_dopler_m

    def read_t_K(self, filepath_t_K):
        """
        Reads temparature profile from .txt file

        Return: t_K, d_T_K, r_t_K_m - radius
        """
        data_temp = read_txt_to_array(filepath_t_K)
        t_K = data_temp[:, 1]
        d_T_K = data_temp[:,2]
        r_t_K_m = data_temp[:, 0]*1e-3
        return t_K, d_T_K, r_t_K_m

    def concentration_n_i(self, d_dopler_lambda_m, kappa_m):
        lambda_сm = self.lambda_m *1e2
        d_dopler_lambda_сm = d_dopler_lambda_m * 1e2
        kappa_cm = kappa_m *1e-2
        koef = 8.19e-20
        n_i_m_3 = d_dopler_lambda_сm * kappa_cm / (lambda_сm**2) / self.f_ik / koef 
        return n_i_m_3


    def concentration_n(self, n_i, stat_sum, g_i, T_K):
        E_i_J = self.E_i_eV 
        return n_i * stat_sum * np.exp(E_i_J / (self.k_B * T_K)) / g_i


    def __stat_sum_read(self, file_path, temperature_value):
        with open(file_path, "r") as file:
            data = file.readlines()
        stat_sum = np.array([float(line.strip()) for line in data])
        return stat_sum[temperature_value]


    def calculate_plasma_parameters(
        self, radius, T_K, d_T_K, kappa_profile_1_m, filepath_statsum
    ):
        """
        n - Cooper number density
        dn - - Cooper number density error
        n_i - population number density
        lambda_broadening_Doplers - Doplers' mechanism of broadening
        
        """
        n = np.empty_like(radius)
        n_i = np.empty_like(radius)
        d_lambda_m = np.empty_like(radius)
        for i, r in enumerate(radius):
            current_T_K = T_K[i]
            stat_sum = self.__stat_sum_read(filepath_statsum, int(current_T_K))
            d_lambda_m[i] = self.delta_lambda_doppler_m(current_T_K)
            n_i[i] = self.concentration_n_i(d_lambda_m[i], kappa_profile_1_m[i])
            n[i] = self.concentration_n(n_i[i], stat_sum, self.g_i, current_T_K)
            
        dn = n * self.E_i_eV*eV*d_T_K /T_K**2/1.38e-23
        return n, dn,  n_i, d_lambda_m