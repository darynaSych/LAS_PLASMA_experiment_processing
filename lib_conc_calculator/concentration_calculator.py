import numpy as np
from units_constants import *
from utilites import *  

class PlasmaValuesCalculator:
    def __init__(self, plasma_parameters, k_B=1.38e-23, eV=1.602e-19):

        self.lambda_m = plasma_parameters.get("lambda_m")
        self.mu_Cu = plasma_parameters.get("mu_Cu")
        self.f_ik = plasma_parameters.get("f_ik")
        self.g_i = plasma_parameters.get("g_i")
        self.E_i = plasma_parameters.get("E_i")
        self.E_k = plasma_parameters.get("E_k")
        self.k_B = k_B


    def delta_lambda_doppler(self,t_K):
        '''
        Return: d_lambda_d
        '''
        d_lambda_d = 7.16e-7 * self.lambda_m * np.sqrt(t_K / self.mu_Cu)
        return d_lambda_d

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

    def concentration_n_i(self, delta_lambda_m, kappa):
        # print(f"kappa: {kappa}")
        return (
            kappa
            * 1e-2
            * delta_lambda_m
            / nm
            / (8.19e-20 * self.f_ik * (self.lambda_m / nm) ** 2)
        )


    def concentration_n(self, n_i, stat_sum, g_i, T_K):
        return n_i * stat_sum / np.exp(-self.E_i / self.k_B / T_K) / g_i


    def __stat_sum_read(self, file_path, temperature_value):
        with open(file_path, "r") as file:
            data = file.readlines()
        stat_sum = np.array([float(line.strip()) for line in data])
        return stat_sum[temperature_value]


    def calculate_plasma_parameters(
        self, radius, T_K, d_T_K, kappa_profile, filepath_statsum
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
            d_lambda_m[i] = self.delta_lambda_doppler(current_T_K)
            n_i[i] = self.concentration_n_i(d_lambda_m[i], kappa_profile[i])
            n[i] = self.concentration_n(n_i[i], stat_sum, self.g_i, current_T_K)
            # print(f'T: {current_T_K}\td_lambda_m: {d_lambda_m[i]}\tn_i: {n_i[i]}\tn: {n[i]}')
            
        dn = n * self.E_i*eV*d_T_K /T_K**2/1.38e-23
        return n, dn,  n_i, d_lambda_m