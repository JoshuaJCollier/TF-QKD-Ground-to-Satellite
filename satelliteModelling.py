# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:01:05 2024

@author: Josh Collier
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from classes import TrigFuncs, Laser, Satellite, Receiver

# ----------------------------------------------------------------- Functions / Classes -----------------------------------------------------------------

def plot_x_y(x, y, xscale='linear', yscale='linear', xaxis='x', yaxis='y'):
    """ Just quick plotting tool for debugging.
    """
    fig, ax = plt.subplots()
    plt.plot(x, y)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_title("{} vs {}".format(yaxis, xaxis))
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    plt.show()
    input('Press enter to continue...')
    plt.close()


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
        windspeed = windspeed[:, :, None]
        phase_psd = 0.016 * self.laser.wavenumber**2 * height_steps[:, :, None] * (1/windspeed) * c2n[:, :, None] * ((self.f[None, None, :]/windspeed)**2 + (1/(2*np.pi*L_0))**2)**(-4/3)
        phase = np.trapz(phase_psd, axis=1)
        return phase
    
    def get_greenwood_tarazano(self, height_steps, c2n, windspeed, L_0):
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
            plt.plot(actual_heights[10], windspeed[10])
            plt.plot(actual_heights[10], c2n[10])
            plt.plot(actual_heights[10], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps))[10])
            plt.plot(actual_heights[10], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps)*c2n*windspeed**(5/3))[10])
            plt.plot(actual_heights[99], windspeed[99])
            plt.plot(actual_heights[99], c2n[99])
            plt.plot(actual_heights[99], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps))[99])
            plt.plot(actual_heights[99], (self.rec.lengthsteps2sat(self.sat, self.no_length_steps)*c2n*windspeed**(5/3))[99])
            
            plt.yscale('log')
            plt.xlabel('Height (m)')
            plt.ylabel('Value')
            plt.title('Whats happening? Vrms(peak)={:.2f}m/s, Vrms(end)={:.2f}m/s'.format(rms_windspeed[10], rms_windspeed[99]))
            plt.legend(['Windspeed (peak)', 'Cn^2 (peak)', 'Length (peak)', 'Kolmogorov comp (peak)', 'Windspeed (end)', 'Cn^2 (end)', 'Length (end)', 'Kolmogorov comp (end)'])
            plt.ylim(1e-22, 1e5)
            plt.xlim(0, 1e5)
            plt.grid()
            #plt.show()
        
        return kolmogorov_phase_psd

# ---------------------------------------------------------------------- Constants ----------------------------------------------------------------------

# Dimensionality of arrays:
no_time_steps = 100
no_freq_steps = 100000
no_length_steps = 100000
no_q_steps = 1
steps_array = [no_time_steps, no_freq_steps, no_length_steps, no_q_steps]
#print("Max array data size: {:.2f}GB".format(no_time_steps*no_length_steps*no_freq_steps*no_q_steps*8/(1024**3)))
plot_atmos = False
plot_PSDs = True

earth_radius = 6371e3 # radius of earth (m)
speed_of_light = 2.99792458*1e8 # (m/s)
refractive_index = 1.00027 # refractive index of the atmosphere, this can probably be changed over distance but also small, will make less than 0.03% diff
ground_station_seperation = 2000e3 # seperation between Alice and Bob (m)
sat_LEO_low = Satellite(500e3, 7.5e3, 60*6, no_time_steps) # height, speed, time of satellite passover -> these are all similar to  GRACE-FO
sat_LEO_high = Satellite(2000e3, 7.0e3, 60*14, no_time_steps) # height, speed, time of satellite passover -> these are all similar to  GRACE-FO
sat_MEO_low = Satellite(10e6, 3.5e3, 60*30, no_time_steps) # height, speed, time of satellite passover -> these are all similar to  GRACE-FO
sat_MEO_high = Satellite(20e6, 3.5e3, 60*45, no_time_steps) # height, speed, time of satellite passover -> these are all similar to  GRACE-FO

#sat_LEO_max = Satellite(1500e3, 7.1e3, 60*18) # height, speed, time of satellite passover
# V(0) = 10 # m/s Ground wind speed ([1] suggests 10, 21, 30), C2n(0) = 1.7e-14 # Andrews, 2009: Near-ground vertical profile of refractive-index fluctuations (C2n(0)) ([1] suggests 1.7*10^-13 or 1.7*10^-14)
alice = Receiver(-ground_station_seperation/(2*earth_radius), V_0=10, C2n_0=1e-14) # these values are used in other satellite paper (Wang I think)
bob = Receiver(ground_station_seperation/(2*earth_radius), V_0=10, C2n_0=1e-14)
quantum_laser = Laser(1550.12e-9) # quantum laser wavelength from [5]
reference_laser = Laser(1548.51e-9) # reference laser wavelength from [5]

# IMPORTANT INPUT PARAMETERS
phase_stab_bandwidth = 1e5
laser_stab_bandwidth = 3e5

total_figures = 1
# --------------------------------------------------------------------- Simulations ---------------------------------------------------------------------

def generate_phase_time_plot():
    global total_figures
    #sat_used = sat_LEO_current # this is an object of type Satellite
    max_time = 1 # for our instance, if we imagine integration times of n seconds, we integrate between 1/n to infinity 
    max_freq_simulated = 1e10 # (where infinity above must be bound here to some large value, which hopefully converges)
    # possibly upper bound will be limited by sampling frequency of device we are doing, with worst case scenario upper bound of freq of light (~10^14)
    freq_space = np.logspace(np.log10(1/max_time), np.log10(max_freq_simulated), no_freq_steps) # note, bound this by the limits of the integral
    
    # ------------------ Laser stuff ------------------ Information from Bertaina paper [2]
    # Cavity PSD
    c4, c3, c2 = 0.5, 0, 2e-3
    S_cavity = (c4/freq_space**4+c3/freq_space**3+c2/freq_space**2)[None, :] # not sure yet - note that this does not converge, so if we add more freq integral, number gets bigger
    # Free running laser and stabilised laser PSD
    B, gamma, delta = laser_stab_bandwidth, 0.1, 10 # B is laser stab bandwidth
    G_0 = (2*np.pi*B)**2 * (1+delta)/(1+gamma)
    G_f = G_0*(1/(2*np.pi*1j*freq_space)**2)*(1j*freq_space+B*gamma)/(1j*freq_space+B*delta)
    r3, r2, fc = 3e6, 3e2, 2e6
    S_laser_free = (r3/(freq_space**3)) + (r2/(freq_space**2)) * (fc/(freq_space+fc))**2
    S_laser_stab = S_cavity + np.abs(1/(1-G_f))**2 * S_laser_free
    
    sat_list = [sat_LEO_low, sat_LEO_high]#, sat_MEO_low, sat_MEO_high]
    sat_name_list = ['LEO low', 'LEO high']#, 'MEO low', 'MEO high']
    S_outputs, sat_times, tot_error_outputs, laser_error_outputs, atmosphere_error_outputs = [], [], [], [], [] # defining some arrays
    
    loop_index = 0
    for sat_used in sat_list:
        # ------------------ Comms stuff ------------------ 
        # from here has the potential of being looped
        to_A = Communication(sat_used, alice, quantum_laser, max_uninterrupt_time=max_time, freq_range=freq_space, steps=steps_array)
        to_B = Communication(sat_used, bob, quantum_laser, max_uninterrupt_time=max_time, freq_range=freq_space, steps=steps_array)
        
        S_AC, S_BC = to_A.generateSim(), to_B.generateSim()
        
        #link_noise = np.maximum(S_AC, S_BC) # characteristic noise of the link
        S_link = S_AC+S_BC # more conservative estimate for S_link
        phase_stab_noise_floor = (reference_laser.wavelength - quantum_laser.wavelength)**2 / quantum_laser.wavelength**2 * (S_link) / freq_space**2
        
        # Path difference and PSD contribution from the laser
        delta_L = alice.dist2sat(sat_used) - bob.dist2sat(sat_used)
        S_contrib_laser = np.sin(2*np.pi*freq_space*refractive_index*delta_L[:, None]/speed_of_light)**2 * S_laser_stab

        # Phase variance without stabilisation (this value is normally unreasonable)
        phase_var_tot = np.trapz(S_link*4+S_contrib_laser*4, freq_space, axis=1) # this is directly from [2] then + 4sin^2(2*pi*f*n*delta_L/c)*ref_laser_noise where delta_L path mismatch abs(AC-BC), n is refractive index
        error = phase_var_tot / 4  # this is QKD QBER from [4]

        # Phase stabilisation transfer function
        bandwidth_freq = phase_stab_bandwidth # phase stab bandwidth 100kHz to 1MHz bandwidth -> takes over laser when less than 1e5, similar to laser at 1e5, dissapears at 1e6 compared to laser (for low LEO)
        phase_stab_transfer_func = 1/(1-1j*bandwidth_freq/freq_space) #G = 2 * np.pi * bandwidth_freq, omega = 2 * np.pi * freq_space, GOL = -1j * G / omega, transfer_func = 1/(1+GOL)
        S_link_phase_stable = np.maximum(S_link*(np.abs(phase_stab_transfer_func)**2), phase_stab_noise_floor)
        
        # Phase variance with stabilisation (our actual output)
        S_tot = S_link_phase_stable*4+S_contrib_laser*4
        phase_var_phase_stable = np.trapz(S_tot, freq_space, axis=1)    
        error_phase_stable = phase_var_phase_stable / 4  # this is QKD QBER from [4]
        
        # Seperate components
        laser_error = np.trapz(S_contrib_laser*4, freq_space, axis=1) / 4
        atmosphere_error = np.trapz(S_link_phase_stable*4, freq_space, axis=1) / 4 
        

        S_outputs.append(S_tot)
        laser_error_outputs.append(laser_error)
        atmosphere_error_outputs.append(atmosphere_error)
        tot_error_outputs.append(error_phase_stable)
        sat_times.append(sat_used.time_array/sat_used.time)
    
        if plot_PSDs:
            plt.figure(total_figures, figsize=(7,7))
            total_figures += 1
            plt.plot(freq_space, S_AC[0])
            #plt.plot(freq_space, S_AC[10])
            #plt.plot(freq_space, S_AC[50])
            #plt.plot(freq_space, S_AC[99])
            plt.plot(freq_space, phase_stab_noise_floor[0])
            plt.plot(freq_space, S_link_phase_stable[0])
            plt.plot(freq_space, S_laser_free)
            plt.plot(freq_space, S_laser_stab[0])
            plt.plot(freq_space, S_tot[0])
        
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase noise (rad^2/Hz)')
            plt.title('Phase PSD at start for {}'.format(sat_name_list[loop_index]))
            plt.legend(['A to C', 'Stab noise floor', 'A to C stabilised', 'Laser Free', 'Laser Stable', 'Total']) #
            plt.grid()
        loop_index += 1

    #plt.legend(sat_name_list)
    
        
    #print("No Phase Stab Mean: {:.2f}\nPhase Stab Mean: {:.2f}".format(np.average(error*100), np.average(error_phase_stable*100)))
    plt.figure(total_figures)
    total_figures += 1
    for i in range(len(tot_error_outputs)):
        plt.plot(sat_times[i], tot_error_outputs[i]*100) 
        plt.plot(sat_times[i], laser_error_outputs[i]*100)
        plt.plot(sat_times[i], atmosphere_error_outputs[i]*100)
    plt.title("QBER vs Time for integration period {}s w/ max freq {:.1E}Hz\n(tstep={}, fstep={}, lstep={})".format(to_A.max_uninterrupt_time, max_freq_simulated, no_time_steps, no_freq_steps, no_length_steps)) # Phase Noise @ 1Hz vs Time
    plt.xlabel("Time (s)") # Time (s) / Freq (Hz)
    plt.ylabel("QBER (%)") # Phase Noise @ 1 Hz (rad^2)
    
    legend_list = []
    for sat_name in sat_name_list:
        legend_list.append(sat_name + ' total')
        legend_list.append(sat_name + ' laser')
        legend_list.append(sat_name + ' atmos')
    plt.legend(legend_list)
    # ax.set_ylim(0, 11), fig.set_size_inches(9, 6)
    
    end = time.time()
    print("Code took {:.2f}s".format(end-start))
    
    plt.show()

def generate_visual_plot():
    sat_used = sat_LEO_low
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