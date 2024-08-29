# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:01:05 2024

@author: Josh Collier
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from classes import TrigFuncs, Laser, Satellite, Receiver, plot_x_y
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14 # 14 for figure 2, 18 for figure 3, 

# ----------------------------------------------------------------- Functions / Classes -----------------------------------------------------------------

class Communication():
    def __init__(self, sat: Satellite, rec: Receiver, laser: Laser, max_uninterrupt_time: float, freq_range, steps):
        """Initialize a communication instance with a specified satellite, receiver and laser.

        Args:
            sat (Satellite): Satellite being communicated with.
            rec (Receiver): Receiver on the ground.
            laser (Laser): Type of laser being used.
            max_uninterrupt_time (float): Maximum time without interupption (this limits the minimum freq)
        """
        self.sat = sat
        self.rec = rec
        self.laser = laser
        self.max_uninterrupt_time = max_uninterrupt_time
        self.f = freq_range[None, :]
        self.no_time_steps = steps[0]
        self.no_length_steps = steps[1]
        self.no_freq_steps = steps[2]
        self.no_q_steps = steps[3]
         
    # from [1]: "Taylors hypothesis fails when V_perp is considerably less than the magnitude of turbulent fluctuations in wind velocities, such as occurs when the mean wind speed is parallel to the line of sight"
    def get_windspeed(self, slew, heights): # note, this windspeed is characterised based on wind speed perpendicular to beam when pointing directly upwards, not when its slanted, may cause issues
        """Get the wind speed at each height step for each time step. Follows the Bufton wind model [1].

        Args:
            slew (arr): 1D array of slew rate of satellite at each time step (in radians/second) 
            heights (arr): 2D numpy array of height values at each height step for each time step (in meters). 

        Returns:
            windspeed (arr): 2D numpy array of windspeed at each height step for each time step (in meters/second).
        """
        windspeed = slew*heights + self.rec.V_0 + 30*np.exp(-np.power((heights-9400)/4800,2))
        #plt.figure(3)
        #plt.plot(windspeed[326])
        #plt.show()
        return windspeed
    
    def get_wind_rms(self, heights, windspeed):
        """Gets the root mean square of wind speed between 5km and 20km [1]. Sometimes known as V_rms.

        Args:
            heights (arr): 2D numpy array of height values at each height step for each time step (in meters). 
            windspeed (arr): 2D numpy array of windspeed at each height step for each time step (in meters/second).

        Returns:
            wind_rms (arr): 1D array of rms windspeed values for each time step (in meters/second).
        """
        height_steps = np.diff(heights, axis=1)
        height_steps = np.concatenate([height_steps, height_steps[:,-1][:, np.newaxis]], axis=1) # makes shape the same, little effect to sim
        
        height_steps_subsection = np.where((heights > 5e3) & (heights < 2e4), height_steps, 0) # this is the subsection of wind speed used for V_rms
        windspeed_subsection = np.where((heights > 5e3) & (heights < 2e4), windspeed, 0) # this is the subsection of wind speed used for V_rms
        windspeed_rms = np.sqrt(np.sum(windspeed_subsection**2 * height_steps_subsection, axis = 1)/(15e3)) 
        return windspeed_rms # np.ones(np.shape(windspeed_rms))*21 
    
    def get_c2n(self, wind_rms, height):
        """Gets the characteristic optical turbulence value (Cn^2) for each height step for each time step. Follows Hufnagel-Valley (H-V) model [1].
        The formal term for Cn^2 is the index of refraction structure constant, sometimes called the structure parameter [1].
        To generate plot: plot_x_y(height, c2n, 'log', 'log', 'Height (m)', 'Cn^2 (m^(-2/3))')

        Args:
            wind_rms (arr): 1D array of rms windspeed values for each time step (in meters/second).
            height (arr): 2D numpy array of height values at each height step for each time step (in meters). 

        Returns:
            c2n (arr): 2D array of optical turbulence value for each height step for each time step (in meters^(-2/3)).
        """
        c2n = 0.00594*np.power(wind_rms[:, np.newaxis]/27, 2) * np.power(1e-5*height, 10) * np.exp(-height/1000) + \
                2.7e-16*np.exp(-height/1500) + \
                self.rec.C2n_0*np.exp(-height/100)
        return c2n
    
    def get_kolmogorov(self, c2n, windspeed):
        """Gets the Kolmogorov phase noise. To generate plot: plot_x_y(self.rec.steps2sat(self.sat)[5, :], phase_psd[5, :, 0], 'log', 'log', 'Height (m)', 'Phase PSD (rad^2/m)')

        Args:
            c2n (arr): 2D array of optical turbulence value for each height step for each time step (in meters^(-2/3)).
            windspeed (arr): 2D numpy array of windspeed at each height step for each time step (in meters/second).
            
        Returns:
            phase_noise_variance (arr) : 1D array of phase noise variance for each time step (in rad^2).
        """
        lengthsteps2sat = self.rec.lengthsteps2sat(self.sat, self.no_length_steps)
        height_integral_component = lengthsteps2sat * c2n * windspeed**(5/3) # This is Kolmogorov noise [3].
        S_f = 0.016*self.laser.wavenumber**2 * np.trapz(height_integral_component, axis=1)[:, None] * self.f**(-8/3) # integrates it over height
        return S_f
    
    def get_von_karman(self, height_steps, c2n, windspeed, L_0):
        # Depricated function
        windspeed = windspeed[:, :, None]
        phase_psd = 0.016 * self.laser.wavenumber**2 * height_steps[:, :, None] * (1/windspeed) * c2n[:, :, None] * ((self.f[None, None, :]/windspeed)**2 + (1/(2*np.pi*L_0))**2)**(-4/3)
        phase = np.trapz(phase_psd, axis=1)
        return phase
    
    def get_greenwood_tarazano(self, height_steps, c2n, windspeed, L_0):
        # Depricated function
        q_min, q_max = 0, 100
        q = np.linspace(q_min, q_max, self.no_q_steps)[None, None, None, :]
        windspeed = windspeed[:, :, None]
        f_on_wind = ((self.f/windspeed)**2)[:, :, :, None]
        integ = np.trapz(((f_on_wind + q**2 + np.sqrt(q**2 + f_on_wind/(2*np.pi*L_0)))**(-11/6)), q, axis=3)
        phase_psd = 0.0097 * self.laser.wavenumber**2 * c2n[:, :, None] * height_steps[:, :, None] * (1/windspeed) * integ
        phase = np.trapz(phase_psd, axis=1)
        return phase
        
    def generateSim(self):
        global total_figures
        # This generates a 2D array, where with one dimesion being time (even steps), other being height (not even steps)
        actual_heights = self.rec.steps2sat(self.sat, self.no_length_steps)

        slew_rate = np.array(self.rec.slew2sat(self.sat))[:, np.newaxis] # rad/s - zenith angle rate of change
        windspeed = self.get_windspeed(slew_rate, actual_heights) # this is the Bufton wind model value of wind speed at each step of height that we are simulating [1]
        # here go through and calculate the angle of the satellite at each point so that I can find the perpenducilar wind speed
        rms_windspeed = self.get_wind_rms(actual_heights, windspeed) # this is the rms of windspeed, used in the calculation of Cn^2 [1] (sometimes assumed to be just 21m/s)
        c2n = self.get_c2n(rms_windspeed, actual_heights) # this is our Hufnagel-Valley turbulence model for Cn^2 [1]
        kolmogorov_phase_psd = self.get_kolmogorov(c2n, windspeed) # this is the phase noise PSD [2, 3]
        #von_karman_phase_noise_var, von_corrected = self.get_von_karman(height_steps, c2n, windspeed, 100) # using outer scale of 100m
        #greenwood_tarazano_phase_noise_var, gre_corrected = self.get_greenwood_tarazano(actual_heights, c2n, windspeed, 10) # using outer scale of 100m
        
        if plot_atmos:
            plt.figure(total_figures, figsize=(7,7))
            total_figures += 1
            plt.plot(actual_heights[int(no_time_steps/10)], windspeed[int(no_time_steps/10)])
            plt.plot(actual_heights[int(no_time_steps/10)], c2n[int(no_time_steps/10)])
            plt.plot(actual_heights[int(no_time_steps/10)], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps))[int(no_time_steps/10)])
            plt.plot(actual_heights[int(no_time_steps/10)], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps)*c2n*windspeed**(5/3))[int(no_time_steps/10)])
            plt.plot(actual_heights[-1], windspeed[-1])
            plt.plot(actual_heights[-1], c2n[-1])
            plt.plot(actual_heights[-1], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps))[-1])
            plt.plot(actual_heights[-1], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps)*c2n*windspeed**(5/3))[-1])
            
            plt.yscale('log')
            plt.xlabel('Height (m)')
            plt.ylabel('Value')
            plt.title('Whats happening? Vrms(peak)={:.2f}m/s, Vrms(end)={:.2f}m/s'.format(rms_windspeed[10], rms_windspeed[-1]))
            plt.legend(['Windspeed (peak)', 'Cn^2 (peak)', 'Length (peak)', 'Kolmogorov comp (peak)', 'Windspeed (end)', 'Cn^2 (end)', 'Length (end)', 'Kolmogorov comp (end)'])
            plt.ylim(1e-22, 1e5)
            plt.xlim(0, 1e5)
            plt.grid()
            #plt.show()
        
        return kolmogorov_phase_psd

# ---------------------------------------------------------------------- Constants ----------------------------------------------------------------------

# Dimensionality of arrays:
no_time_steps = 1001
no_freq_steps = 1000
no_length_steps = 1000
no_q_steps = 1
steps_array = [no_time_steps, no_freq_steps, no_length_steps, no_q_steps]
#print("Max array data size: {:.2f}GB".format(no_time_steps*no_length_steps*no_freq_steps*no_q_steps*8/(1024**3)))
plot_atmos = False
plot_PSDs = False
plot_qber = True
#plot_all_on_one = False
earth_radius = 6371e3 # radius of earth (m)
speed_of_light = 2.99792458*1e8 # (m/s)
refractive_index = 1.00027 # refractive index of the atmosphere, this can probably be changed over distance but also small, will make less than 0.03% diff

#sat_LEO_max = Satellite(1500e3, 7.1e3, 60*18) # height, speed, time of satellite passover
# V(0) = 10 # m/s Ground wind speed ([1] suggests 10, 21, 30), C2n(0) = 1.7e-14 # Andrews, 2009: Near-ground vertical profile of refractive-index fluctuations (C2n(0)) ([1] suggests 1.7*10^-13 or 1.7*10^-14)
quantum_laser = Laser(1550.12e-9) # quantum laser wavelength from [5]
reference_laser = Laser(1548.51e-9) # reference laser wavelength from [5]

# IMPORTANT INPUT PARAMETERS
phase_stab_bandwidth = 1e5
laser_stab_bandwidth = 3e5

total_figures = 1
# --------------------------------------------------------------------- Simulations ---------------------------------------------------------------------

def generate_phase_time_plot():
    # ------------------ Set up stuff ------------------
    global total_figures
    #max_time = 1 # for our instance, if we imagine integration times of n seconds, we integrate between 1/n to infinity 
    
    max_time_array = [0.1, 10]
    num_max_times_plotted = 1
    seperation_array = [1000e3, 2000e3]#1500e3, 2000e3]
    sat_array = [500e3, 10e6]#, 2000e3, 10e6] #sat_LEO_low, sat_LEO_high, sat_MEO_low]#, sat_MEO_low, sat_MEO_high]
    sat_name_list = ['LEO', 'MEO']#, 'MEO low']#, 'MEO low', 'MEO high']
        
    fig, axs = plt.subplots(len(sat_array), len(seperation_array), sharex=False, sharey='row')

    total_figures += 1
    # fig.suptitle('QBER vs Time') # we dont need a title for the paper
    fig.supxlabel('Time (min)')
    fig.supylabel('QBER (%)')
    plt_labels = [['(a)', '(b)'], ['(c)', '(d)']]
    
    # Legend list for if you are using multiple tau
    legend_list = []
    for i in range(num_max_times_plotted):
        legend_list.append('$\\tau_i$={:.2f}s stable'.format(max_time_array[i]))
    
    print('Starting...')
    for sep_index in range(len(seperation_array)):
        axs[sep_index][0].set_yticks([0.0, 0.5, 1.0, 1.5])
        axs[sep_index][0].set_ylim([0.0, 1.5])
        for sat_index in range(len(sat_array)):
            loop_start = time.time()
            S_outputs, sat_times, tot_error_outputs, laser_error_outputs, atmosphere_error_outputs, tot_unstable_error_outputs = [], [], [], [], [], [] # defining some arrays
            for time_index in range(len(max_time_array)):
                ground_station_seperation = seperation_array[sep_index]
                sat_used = Satellite(sat_array[sat_index], ground_station_seperation, 17*np.pi/36, no_time_steps)
                max_time = max_time_array[time_index]
                alice = Receiver(-ground_station_seperation/(2*earth_radius), V_0=10, C2n_0=1e-14) # these values are used in other satellite paper (Wang I think)
                bob = Receiver(ground_station_seperation/(2*earth_radius), V_0=10, C2n_0=1e-14)

                max_freq_simulated = 1e9 # (where infinity above must be bound here to some large value, which hopefully converges)
                freq_space = np.logspace(np.log10(1/max_time), np.log10(max_freq_simulated), no_freq_steps) # note, bound this by the limits of the integral
                multiplication_factor = 4 # conservative estimate from Bertania
                
                # ------------------ Laser stuff ------------------ Information from Bertaina paper [2]
                # Cavity PSD
                c4, c3, c2 = 0.5, 0, 2e-3
                S_cavity = (c4/freq_space**4+c3/freq_space**3+c2/freq_space**2)[None, :] # not sure yet
                
                # Free running laser and stabilised laser PSD
                B, gamma, delta = laser_stab_bandwidth, 0.1, 10 # B is laser stab bandwidth
                G_0 = (2*np.pi*B)**2 * (1+delta)/(1+gamma)
                G_f = G_0*(1/(2*np.pi*1j*freq_space)**2)*(1j*freq_space+B*gamma)/(1j*freq_space+B*delta)
                r3, r2, fc = 3e6, 3e2, 2e6
                S_laser_free = (r3/(freq_space**3)) + (r2/(freq_space**2)) * (fc/(freq_space+fc))**2
                S_laser_stab_old = S_cavity + np.abs(1/(1-G_f))**2 * S_laser_free
                
                # NEW LASER STABILISED FROM GRACE-FO DATA
                log_f = np.log10(freq_space)                
                grace_FO_rees2021 = 0.001*log_f**5 + 0.0046*log_f**4 - 0.0422*log_f**3 - 0.0683*log_f**2 - 1.4827*log_f + 0.3217
                S_laser_stab = [np.where(freq_space < 10, 10**grace_FO_rees2021, 0.0542*10**1/freq_space**1)**2]
                #plot_x_y(freq_space, [S_laser_stab_old[0], S_laser_stab[0]], 'log', 'log')
                
                # ------------------ Comms stuff ------------------ 
                # from here has the potential of being looped
                to_A = Communication(sat_used, alice, quantum_laser, max_uninterrupt_time=max_time, freq_range=freq_space, steps=steps_array)
                to_B = Communication(sat_used, bob, quantum_laser, max_uninterrupt_time=max_time, freq_range=freq_space, steps=steps_array)
                
                S_AC, S_BC = to_A.generateSim(), to_B.generateSim()
                
                #link_noise = np.maximum(S_AC, S_BC) # characteristic noise of the link
                S_link = S_AC+S_BC # more conservative estimate for S_link
                phase_stab_noise_floor = (reference_laser.wavelength - quantum_laser.wavelength)**2 / quantum_laser.wavelength**2 * (S_link) / freq_space**2
                
                # Path difference and PSD contribution from the laser
                delta_L = np.abs(alice.dist2sat(sat_used) - bob.dist2sat(sat_used))
                #delta_L = delta_L*0
                S_contrib_laser = np.sin(2*np.pi*freq_space*refractive_index*delta_L[:, None]/speed_of_light)**2 * S_laser_stab
                
                # Phase variance without stabilisation (this value is normally unreasonable)
                S_tot_unstable = S_link*multiplication_factor+S_contrib_laser*multiplication_factor
                phase_var_tot_unstable = np.trapz(S_tot_unstable, freq_space, axis=1) # this is directly from [2] then + 4sin^2(2*pi*f*n*delta_L/c)*ref_laser_noise where delta_L path mismatch abs(AC-BC), n is refractive index
                error_unstable = phase_var_tot_unstable / 4  # this is QKD QBER from [4]

                # Phase stabilisation transfer function
                bandwidth_freq = phase_stab_bandwidth # phase stab bandwidth 100kHz to 1MHz bandwidth -> takes over laser when less than 1e5, similar to laser at 1e5, dissapears at 1e6 compared to laser (for low LEO)
                phase_stab_transfer_func = 1/(1-1j*bandwidth_freq/freq_space) #G = 2 * np.pi * bandwidth_freq, omega = 2 * np.pi * freq_space, GOL = -1j * G / omega, transfer_func = 1/(1+GOL)
                S_link_phase_stable = np.maximum(S_link*(np.abs(phase_stab_transfer_func)**2), phase_stab_noise_floor)
                
                # Phase variance with stabilisation (our actual output)
                S_tot = S_link_phase_stable*multiplication_factor+S_contrib_laser*multiplication_factor
                phase_var_phase_stable = np.trapz(S_tot, freq_space, axis=1)
                
                if ((sep_index == 0) and (sat_index == 0) and (time_index == (len(max_time_array)-1))) and plot_qber:
                    min_freq_arr = np.logspace(-1,6,100)
                    #print(min_freq_arr)
                    phase_var_unstable_at_freq = []
                    phase_var_phase_stable_at_freq = []
                    phase_var_laser_at_freq = []
                    phase_var_link_at_freq = []
                    for min_freq in min_freq_arr:
                        freq_space_inst = np.where(freq_space >= min_freq, freq_space, 0)
                        phase_var_unstable_at_freq.append(np.mean(np.trapz(S_tot_unstable, freq_space_inst, axis=1)))
                        phase_var_phase_stable_at_freq.append(np.mean(np.trapz(S_tot, freq_space_inst, axis=1)))
                        phase_var_laser_at_freq.append(np.mean(np.trapz(S_contrib_laser*multiplication_factor, freq_space_inst, axis=1)))
                        phase_var_link_at_freq.append(np.mean(np.trapz(S_link_phase_stable*multiplication_factor, freq_space_inst, axis=1)))
                    
                    fig_Q, ax1_Q = plt.subplots()
                    total_figures += 1
                    ax2_Q = ax1_Q.twinx() 
                    ax1_Q.plot(1/min_freq_arr, np.array(phase_var_unstable_at_freq)*25) # to get QBER we multiply by 100% and divide by 4 so we get 25
                    ax1_Q.plot(1/min_freq_arr, np.array(phase_var_phase_stable_at_freq)*25)
                    ax1_Q.plot(1/min_freq_arr, np.array(phase_var_laser_at_freq)*25)
                    ax1_Q.plot(1/min_freq_arr, np.array(phase_var_link_at_freq)*25)
                    ax1_Q.legend(['Unstable Link', 'Stablised Link', 'Laser Contribution', 'Atmospheric Contribution'])
                    ax1_Q.set_yscale('log')
                    ax1_Q.set_ylim(0.0001,50)
                    ax1_Q.set_ylabel('QBER (%)')
                    ax1_Q.set_xscale('log')
                    ax1_Q.set_xlim(1e-6,4)
                    ax1_Q.set_xlabel('Integration time (s)')
                    ax2_Q.set_yscale('log')
                    ax2_Q.set_ylim(0.0004,200)
                    ax2_Q.set_ylabel('Phase noise $\\sigma_\\phi^2$ (rad$^2$)')
                    #plt.savefig(r'C:\Users\josh\OneDrive - UWA\UWA\PhD\6. Photos\SatQKD\QBERPhaseNoiseIntegTime.png', bbox_inches='tight')
                    #plt.title('QBER / Phase noise vs Integration time') # dont need title for paper figures
                        
                error_phase_stable = phase_var_phase_stable / 4  # this is QKD QBER from [4]
                
                # Seperate components
                laser_error = np.trapz(S_contrib_laser*multiplication_factor, freq_space, axis=1) / 4
                atmosphere_error = np.trapz(S_link_phase_stable*multiplication_factor, freq_space, axis=1) / 4 
                
                S_outputs.append(S_tot)
                laser_error_outputs.append(laser_error)
                atmosphere_error_outputs.append(atmosphere_error)
                tot_unstable_error_outputs.append(error_unstable)
                tot_error_outputs.append(error_phase_stable)
                sat_times.append(sat_used.time_array)
            
                if plot_PSDs and (time_index == (len(max_time_array)-1)) and (sep_index == 0) and (sat_index == 0):
                    #plt.figure(total_figures, figsize=(7,7))
                    fig, axes = plt.subplots(1, 2, sharex=False, sharey=True)
                    total_figures += 1
                    
                    steps = [0, int(no_time_steps/2)+1]
                    step_names = ['Start', 'Middle']
                    label_plt = ['(a)', '(b)']
                
                    fig.supxlabel('Frequency (Hz)')
                    fig.supylabel('Phase noise (rad$^2$/Hz)')
                    
                    for i in range(len(axes)):
                        ax = axes[i]
                        step = steps[i]
                        step_name = step_names[i]
                        
                        print('$\\Delta L$ =', delta_L[step])
                        print('$L_\{AC\}$ =', alice.dist2sat(sat_used)[step])
                        print('$\\phi_{AC}$ =', np.abs(np.arctan((alice.y-sat_used.y[step])/(alice.x-sat_used.x[step]))) + alice.angle)
                        print('Phase PSD at start for {} with {:.0f}km sep'.format(sat_name_list[sat_index], ground_station_seperation/1000))
                    
                        ax.plot(freq_space, S_link_phase_stable[step]*4)
                        ax.plot(freq_space, S_contrib_laser[step]*4)
                        ax.plot(freq_space, S_tot[step])
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_ylim(1e-11, 1e-2)
                        ax.set_yticks([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])#, 1e-1, 1e0, 1e1, 1e2, 1e3])
                        ax.set_xlim(1e0,1e6)
                        ax.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
                        ax.grid()
                        ax.legend(['{} atmospheric noise'.format(step_name), '{} laser noise'.format(step_name), '{} total'.format(step_name)]) #
                        ax.annotate(
                            label_plt[i],
                            xy=(0, 0), xycoords='axes fraction',
                            xytext=(+0.3, +1.3), textcoords='offset fontsize', verticalalignment='top')
                        
                    #plt.title('Phase PSD at start for {} with {:.0f}km sep'.format(sat_name_list[sat_index], ground_station_seperation/1000))
                        
            #print("No Phase Stab Mean: {:.2f}\nPhase Stab Mean: {:.2f}".format(np.average(error*100), np.average(error_phase_stable*100)))
            ax = axs[sat_index][sep_index]
            
            ax.annotate(
                plt_labels[sat_index][sep_index],
                xy=(0, 0), xycoords='axes fraction',
                xytext=(+0.3, +1.3), textcoords='offset fontsize', verticalalignment='top')
            
            for i in range(num_max_times_plotted): #len(tot_error_outputs)):
                plot_output = tot_error_outputs[i][1:-1]*100 #np.where(tot_error_outputs[i][:-1] < 1, tot_error_outputs[i][:-1]*100, 100)
                ax.plot(sat_times[i][1:-1]/60, plot_output)
            #plt.title("QBER vs Time for integration period {}s w/ max freq {:.1E}Hz\n(tstep={}, fstep={}, lstep={})".format(to_A.max_uninterrupt_time, max_freq_simulated, no_time_steps, no_freq_steps, no_length_steps)) # Phase Noise @ 1Hz vs Time
            
            if sep_index == (len(seperation_array)-1):
                ax.yaxis.set_label_position("right")
                ax.set_ylabel("{} ({:.0f}km)".format(sat_name_list[sat_index], sat_used.height/1000))
            
            if sat_index == 0:
                ax.set_title("{:.0f}km seperation".format(ground_station_seperation/1000))
            
            end = time.time()
            #last_end = end
            print("Time of {:.0f}km satellite passover for {:.0f}km seperation: {:.2f}s".format(sat_used.height/1000, ground_station_seperation/1000, sat_used.time))
            print("Satellite speed: {:.2f}km/s, {:.6f}rad/s".format(sat_used.speed_m/1000, sat_used.speed))
            print("Iteration took {:.2f}s\n".format(end-loop_start))
    print("Code took {:.2f} mins".format((end-start)/60))
    
    if num_max_times_plotted > 1:
        fig.legend(legend_list, loc='upper right', title="Integration time")
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    

def generate_visual_plot(sat_used: Satellite):
    ground_station_seperation = 2000e3
    alice = Receiver(-ground_station_seperation/(2*earth_radius), V_0=10, C2n_0=1e-14) # these values are used in other satellite paper (Wang I think)
    bob = Receiver(ground_station_seperation/(2*earth_radius), V_0=10, C2n_0=1e-14)

    #sat_used = sat_LEO_low
    fig, ax = plt.subplots()
    divisor = 1000
    # Tangent lines r_A_perp, a_A_perp, r_B_perp, a_B_perp = radial_tangent(alice.radius, alice.angle), radial_tangent(bob.radius, bob.angle)
    y, x = TrigFuncs.pol2cart(earth_radius*np.ones(no_time_steps)/divisor, np.linspace(0,2*np.pi,no_time_steps))
    ax.plot(x, y)
    y, x = TrigFuncs.pol2cart(alice.radius*np.ones(no_time_steps)/divisor, alice.angle)
    ax.plot(x, y,'ro')
    y, x = TrigFuncs.pol2cart(bob.radius*np.ones(no_time_steps)/divisor, bob.angle)
    ax.plot(x, y,'go')
    ax.set_xlim(-4.5e6/divisor, 4.5e6/divisor)
    ax.set_ylim(5e6/divisor, 8e6/divisor)
    ax.set_title("Model of LEO Satellite Path Around Earth")
    ax.set_xlabel("Distance (km)")
    
    fig.set_size_inches(10, 3.33)
    
    artists = []
    y, x = TrigFuncs.pol2cart(sat_used.radius/divisor, sat_used.angle)
    for i in range(200):
        container, = ax.plot([sat_used.y[int(no_time_steps/200*i)]/divisor, alice.y/divisor], [sat_used.x[int(no_time_steps/200*i)]/divisor, alice.x/divisor], color='red')
        container2, = ax.plot([sat_used.y[int(no_time_steps/200*i)]/divisor, bob.y/divisor], [sat_used.x[int(no_time_steps/200*i)]/divisor, bob.x/divisor], color='green')
        container3, = ax.plot(x[int(no_time_steps/200*i)], y[int(no_time_steps/200*i)], 'bo')
        artists.append([container, container2, container3])
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=30)
    ani.save(filename=r"C:\Users\22503577\OneDrive - UWA\UWA\PhD\7. Code\tmp\pillow_example.gif", writer="pillow")
    plt.show()

start = time.time()

#generate_visual_plot()
generate_phase_time_plot()

plt.show()


'''
References:
[1] Laser Beam Propagation through Random Media. Andrews, 2005
[2] Phase Noise in Real-World Twin-Field Quantum Key Distribution. Bertaina, 2023
[3] Optical timing jitter due to atmospheric turbulence: comparison of frequency comb measurements to predictions from micrometeorological sensors. Cladwell, 2020
[4] Coherent phase transfer for real-world twin-ﬁeld quantum key distribution. Clivati, 2022
[5] 600-km repeater-like quantum commuinications with dual-band stabilization. Pitaluga, 2021
'''



'''
# Dead code
LEO_used = LEO_max_arr
xC, yC = pol2cart(LEO_used['r'], LEO_used['a'])
a_len = np.sqrt((xC-xA)**2 + (yC-yA)**2)
a_path_height = np.arctan2((xC-xA), (yC-yA))
h = np.linspace(0, LEO_used['h'], steps)
hstep = LEO_used['h'] / steps
f = np.linspace(1e-2, 1e5, steps)
slew_rate_A = np.append([9e-5],np.diff(np.arctan2(yC-yA, xC-xA))) # this value seems too low
V_A = getWindspeed(h, Vg, slew_rate_A)
C2n_A = getC2n_curve(h, V_A, slew_rate_A, C2n_0, hstep)
sat_pos = [xA[0], yA[0]]
phaseVal_A = getPhaseNoise(k, C2n_A, a_len[5000], V_A, f, h, hstep)

# This gives plot of distance between C and A and C and B over time, and the difference between those path lengths
#xA, yA = alice.x, alice.y
#xB, yB = bob.x, bob.y


def getWindspeed(h, Vg, slew_rate):
    # this is used to calculate a 
    return slew_rate*h + Vg + 30*np.exp(-np.power((h-9400)/4800,2)) # Wind speed (modelled by Bufton wind model)

def getC2n_curve(h, V, C2n_0, hstep):
    w = np.sqrt((1/15e3)*np.trapz(V[int(5e3/hstep):int(20e3/hstep)]**2)*hstep)
    C2n = 0.00594*np.power(w/27, 2) * np.power(1e-5*h, 10) * np.exp(-h/1000) + \
            2.7e-16*np.exp(-h/1500) + \
            C2n_0*np.exp(-h/100) # this should give us C2n as a function of height and wind speed
    return C2n

#Var[α→] = 1.093*L*C2n*D_r**(−1/3) # other equation from bens paper


# find a relation between C2n and phase stability in davids paper
# use relation of phase stability and QBER to define how well QKD would work
# phase_variance = C2n # some mathematical relation hopefully teehee
# e = phase_variance / 4
def getPhaseNoise(k, C2n, L, V, hstep, dist): # Kolmogorov turbulence
    #C2n and V are arrays
    # 0.033*k**2 * C2n * L * V**(5/3) * f**(-8/3) # https://opg.optica.org/view_article.cfm?pdfKey=4143022c-db7d-4cf3-bbcf1dd4329761be_540926
    
    phaseNoiseProfile = 0
    for i in range(len(L)): # each segment of length steps
        phaseNoiseProfile += 0.033*k**2 * C2n[int(L[i]/hstep)] * dist/hstep * V[i]**(5/3)
    return phaseNoiseProfile # 60000 # np.trapz(f**(-8/3)) from 0.001 to 10000 Hz np.trapz(phaseNoiseProfile)

def generate_phase_time_plot():
    fig, ax = plt.subplots()#subplot_kw={'projection': 'polar'})
    
    sat_array = [sat_LEO_min, sat_LEO_max] # this is an object of type Satellite
    sat_name = ['LEO min', 'LEO max']
    i = 0
    for sat_used in sat_array:
        start = time.time()
        a_len = alice.dist2sat(sat_used)
        b_len = bob.dist2sat(sat_used)
        
        slew_AC = alice.slew2sat(sat_used)
        slew_BC = bob.slew2sat(sat_used)
        phase_A = []
        phase_B = []
        
        vertical_steps = 1000
        
        h = np.arange(0, sat_used.height, step=vertical_steps) # array of heights, divided into steps up to satellite height
        hstep = h[1] # was LEO_used['h'] / steps 
        for j in range(len(sat_used.time_array)):
            windSpeed_A = getWindspeed(h, Vg, slew_AC[j]) # this is an array
            windSpeed_B = getWindspeed(h, Vg, slew_BC[j]) # this is an array
            C2n_A = getC2n_curve(h, windSpeed_A, C2n_0, hstep) # this is an array
            C2n_B = getC2n_curve(h, windSpeed_B, C2n_0, hstep) # this is an array            
            phaseVal_A = getPhaseNoise(laser.wavenumber, C2n_A, alice.oldsteps2sat(sat_used, j, vertical_steps), windSpeed_A, hstep, a_len[j])
            phaseVal_B = getPhaseNoise(laser.wavenumber, C2n_B, bob.oldsteps2sat(sat_used, j, vertical_steps), windSpeed_B, hstep, b_len[j])
            phase_A.append(phaseVal_A)
            phase_B.append(phaseVal_B)
        
        end = time.time()
        print("{} took {}s".format(sat_name[i], start-end))
        ax.plot(sat_used.time_array[1:], phase_A[1:])
        ax.plot(sat_used.time_array[1:], phase_B[1:])
        i += 1
    
    sites = ['A', 'B']
    ax.legend(["{} to C, {}km LEO, {} steps".format(site, sat_used.height/1000, steps) for sat_used in sat_array for site in sites])
    #ax.plot(timesteps[1:], phase2[1:])
    ax.set_title("Phase Noise @ 1Hz vs Time") # vs Time
    ax.set_xlabel("Time (s)") # Time (s) / Freq (Hz)
    ax.set_ylabel("Phase Noise @ 1Hz (rad^2)")
    ax.set_yscale('log')
    fig.set_size_inches(9, 6)
    
    plt.show()
    
    def generate_c2n_plot():
    fig, ax = plt.subplots()
    h = np.linspace(0, 2e5, steps)
    optionsVg = [10, 21, 30]
    optionsC2n_0 = [1.7*10**(-13), 1.7*10**(-14)]
    for Vg in optionsVg:
        for C2n_0 in optionsC2n_0:
            V = getWindspeed(h, Vg, 2.7778e-5)
            C2n = getC2n_curve(h, V, 2.7778e-5, C2n_0, 2e5/steps) #2.7778e-5 is degrees of 0.1 arcseconds (slow LEO)   
            ax.plot(h, C2n)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1, 1e5])
    ax.set_ylim([1e-20, 1e-11])
    ax.set_title("Hufnagel-Valley turbulence model Cn^2")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Cn^2")
    ax.legend(["V(0)={}m/s, Cn^2(0)={}".format(Vg, C2n_0) for Vg in optionsVg for C2n_0 in optionsC2n_0 ])
    plt.show()
    
    def oldsteps2sat(self, satellite, time, hsteps):
        xstep = (satellite.x[time] - self.x) / hsteps
        ystep = (satellite.y[time] - self.y) / hsteps
        mag = magnitude(self.x + xstep * np.arange(hsteps), self.y + ystep * np.arange(hsteps)) - earth_radius
        return mag
    '''