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
    
    def delta_doppler_uncertainty(self, lambda_m, t_K, delta_t_K):
        """
        Обчислює похибку Δ(Δλ_D) доплерівського уширення.
        
        Parameters:
        lambda_m : float
            Довжина хвилі в метрах
        t_K : float
            Температура в Кельвінах
        delta_t : float
            Похибка температури в Кельвінах
        mu_Cu : float
            Молекулярна маса (атомна одиниця маси)
        
        Returns:
        float
            Похибка Δ(Δλ_D) у метрах
        """
        lambda_cm = lambda_m * 1e2
        coefficient = 7.16e-7
        partial_derivative = coefficient * lambda_cm / (2 * np.sqrt(t_K * self.mu_Cu))
        delta_lambda_d_cm = partial_derivative * delta_t_K
        return delta_lambda_d_cm * 1e-2  # повертаємо в метрах


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
   
    
    def concentration_n_i_(self, d_dopler_lambda_m, kappa_m):
        lambda_cm = self.lambda_m * 1e9
        d_dopler_lambda_cm = d_dopler_lambda_m *1e9
        kappa_cm = kappa_m *1e-2
        koef = 8.85e-20
        n_i_m_3 = 1e6* d_dopler_lambda_cm * kappa_cm / (lambda_cm**2) / self.f_ik / koef 
        return n_i_m_3
    
    def concentration_n_i(self, d_dopler_lambda_m, kappa_m):
        lambda_cm = self.lambda_m * 1e9
        d_dopler_lambda_cm = d_dopler_lambda_m *1e9
        kappa_cm = kappa_m *1e-2
        koef = 8.19e-20
        n_i_m_3 =  1e6*d_dopler_lambda_cm * kappa_cm / (lambda_cm**2) / self.f_ik / koef 
        return n_i_m_3
    
    def concentration_n_i_error(self, n_i, d_dopler_lambda_m, delta_dopler_lambda_m, kappa_m, delta_kappa_m):
        """
        Обчислює абсолютну похибку Δn_i для заданої концентрації n_i.

        Parameters:
        n_i : float Значення концентрації n_i (м^-3)
        d_dopler_lambda_m : float Δλ_D (доплерівське уширення) в метрах
        delta_dopler_lambda_m : float Похибка Δλ_D в метрах
        kappa_m : float  Коефіцієнт поглинання в м^-1
        delta_kappa_m : float Похибка коефіцієнта поглинання в м^-1

        Returns:
        float  Абсолютна похибка Δn_i
        """
        rel_err_lambda = delta_dopler_lambda_m / d_dopler_lambda_m
        rel_err_kappa = delta_kappa_m / kappa_m
        total_rel_err = np.sqrt(rel_err_lambda**2 + rel_err_kappa**2)
        return n_i * total_rel_err


    def concentration_n_i_(self, d_dopler_lambda_m, kappa_m):
        koef = 8.3e-15
        n_i_m_3 = d_dopler_lambda_m * kappa_m / (self.lambda_m**2) / self.f_ik / koef 
        return n_i_m_3
    
    def concentration_n(self, n_i, stat_sum, g_i, T_K):
        E_i_J = self.E_i_eV * eV
        return n_i * stat_sum * np.exp(E_i_J / (self.k_B * T_K)) / g_i

    def concentration_n_error(self, n, n_i, delta_n_i, T_K, delta_T_K,  k_B=1.38e-23, eV=1.602e-19):
        """
        Обчислює абсолютну похибку Δn (концентрації атомів Cu)
        
        Parameters:
        n : float or np.ndarray            Обчислене значення n
        n_i : float or np.ndarray            Значення n_i
        delta_n_i : float or np.ndarray            Похибка n_i
        T_K : float or np.ndarray            Температура в К
        delta_T_K : float or np.ndarray            Похибка температури
        E_i_eV : float            Енергія збудження в еВ
        
        Returns:
        float or np.ndarray            Абсолютна похибка n
        """
        E_i_J = self.E_i_eV * eV
        rel_err_n_i = delta_n_i / n_i
        term_T = (E_i_J / (k_B * T_K**2)) * delta_T_K
        total_rel_error = np.sqrt(rel_err_n_i**2 + term_T**2)
        return n * total_rel_error


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
        d_lambda_uncertainty_m = np.empty_like(radius)
        for i, r in enumerate(radius):
            current_T_K = T_K[i]
            current_dT_K = d_T_K[i]
            stat_sum = self.__stat_sum_read(filepath_statsum, int(current_T_K))
            d_lambda_m[i] = self.delta_lambda_doppler_m(current_T_K)
            d_lambda_uncertainty_m[i] = self.delta_doppler_uncertainty(lambda_m = self.lambda_m, t_K = current_T_K, delta_t_K = current_dT_K )

            n_i[i] = self.concentration_n_i(d_lambda_m[i], kappa_profile_1_m[i])
            n[i] = self.concentration_n(n_i[i], stat_sum, self.g_i, current_T_K)
        
        n_i_error = self.concentration_n_i_error(n_i=n_i, d_dopler_lambda_m = d_lambda_m, delta_dopler_lambda_m = d_lambda_uncertainty_m, kappa_m = kappa_profile_1_m, delta_kappa_m = kappa_profile_1_m*1e-2)
        dn = self.concentration_n_error(n=n, n_i=n_i, delta_n_i=n_i_error, T_K=T_K, delta_T_K = d_T_K)
        return n, dn,  n_i, n_i_error, d_lambda_m, d_lambda_uncertainty_m
    
    def kappa_from_n_i(self, n_i,d_lambda_m):
        kappa = np.sqrt(np.log(2)/np.pi)*self.lambda_m**2*(1.6e-19)**2*n_i*self.f_ik/d_lambda_m/2/9.1e-31/9e16/8.85e-12
        return kappa