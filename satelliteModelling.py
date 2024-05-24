# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:01:05 2024

@author: 22503577
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
#import scipy.integrate as integrate

# Constants
earth_radius = 6371e3 # radius of earth (m)
steps = 100 # no units (fractional time steps) -> the time steps dont seem to have an effect on the system outcome, only simulation time

class TrigFuncs():
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi

    def magnitude(x, y):
        return np.sqrt(x**2 + y**2)

    def radial_tangent(rho, phi):
        tangentSeries = np.arange(-steps/2, steps/2) # array of values to put in to calculate tangent line
        x, y = TrigFuncs.pol2cart(rho, phi)
        x2 = x + np.cos(phi+np.pi/2)*tangentSeries*1e6
        y2 = y + np.sin(phi+np.pi/2)*tangentSeries*1e6
        r, a = TrigFuncs.cart2pol(x2, y2)
        return r, a

class Laser():
    def __init__(self, wavelength: float):
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi/self.wavelength

class Satellite():
    def __init__(self, height: float, speed: float, time: float):
        self.height = height
        self.speed = speed
        self.time = time
        self.time_array = np.linspace(0, self.time, steps)
        self.radius = self.height + earth_radius
        self.angle = self.speed/self.radius * (self.time_array-self.time/2)
        self.x, self.y = TrigFuncs.pol2cart(self.radius, self.angle) # x and y coordiantes during a timesweep

class Receiver():
    def __init__(self, angle: float, V_0: float, C2n_0: float):
        """_summary_

        Args:
            angle (_type_): _description_
            V_0 (float): Wind speed value at ground, 0m elevation (V(0)) (in meters/second).
            C2n_0 (float): Optical turbulence value at ground, 0m elevation (Cn^2(0)) (in meters^(-2/3)).
        """
        self.angle = angle
        self.V_0 = V_0
        self.C2n_0 = C2n_0
        self.radius = earth_radius
        self.x, self.y = TrigFuncs.pol2cart(self.radius, self.angle)
    
    def dist2sat(self, satellite: Satellite):
        return TrigFuncs.magnitude(satellite.x - self.x, satellite.y - self.y)
    
    def steps2sat(self, satellite: Satellite, hsteps):
        xstep = (satellite.x - self.x) / hsteps
        ystep = (satellite.y - self.y) / hsteps
        return TrigFuncs.magnitude(self.x + xstep[:, np.newaxis] * np.arange(hsteps), self.y + ystep[:, np.newaxis] * np.arange(hsteps)) - earth_radius
    
    def slew2sat(self, satellite: Satellite):
        """ Slew rate of the satellite from the perspective of the receiver.

        Args:
            satellite (Satellite): The satellite that is passing overhead.

        Returns:
            slew_rate (arr): 1D array of slew rate of satellite at each time step (in radians/second).
        """        
        ret_arr = np.diff(np.arctan2(satellite.y-self.y, satellite.x-self.x))/(satellite.time/steps)
        return np.concatenate([ret_arr, [ret_arr[-1]]]) # arc tan will return the value in radians per second

class Communication():
    def __init__(self, sat: Satellite, rec: Receiver, laser: Laser, max_uninterrupt_time: float):
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
    
    def get_windspeed(self, slew, heights):
        """Get the wind speed at each height step for each time step. Follows the Bufton wind model [1].

        Args:
            slew (arr): 1D array of slew rate of satellite at each time step (in radians/second) 
            heights (arr): 2D numpy array of height values at each height step for each time step (in meters). 

        Returns:
            windspeed (arr): 2D numpy array of windspeed at each height step for each time step (in meters/second).
        """
        return slew*heights + self.rec.V_0 + 30*np.exp(-np.power((heights-9400)/4800,2))
    
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
        return np.sqrt(np.sum(windspeed_subsection**2 * height_steps_subsection, axis = 1)/(15e3)) 
    
    def get_c2n(self, wind_rms, height):
        """Gets the characteristic optical turbulence value (Cn^2) for each height step for each time step. Follows Hufnagel-Valley (H-V) model [1].

        Args:
            wind_rms (arr): 1D array of rms windspeed values for each time step (in meters/second).
            height (arr): 2D numpy array of height values at each height step for each time step (in meters). 

        Returns:
            c2n (arr): 2D array of optical turbulence value for each height step for each time step (in meters^(-2/3)).
        """        
        return 0.00594*np.power(wind_rms[:, np.newaxis]/27, 2) * np.power(1e-5*height, 10) * np.exp(-height/1000) + \
                2.7e-16*np.exp(-height/1500) + \
                self.rec.C2n_0*np.exp(-height/100)
    
    def get_phase_noise_var(self, c2n, windspeed):
        """Gets the phase noise 

        Args:
            c2n (arr): 2D array of optical turbulence value for each height step for each time step (in meters^(-2/3)).
            windspeed (arr): 2D numpy array of windspeed at each height step for each time step (in meters/second).

        Returns:
            phase_noise_variance (arr) : 1D array of phase noise variance for each time step (in rad^2).
        """
        length_step = self.rec.dist2sat(self.sat)[:, np.newaxis]
        freq_integral = 3 / (5*(1/self.max_uninterrupt_time)**(5/3)) # This is a simplification of the integral of f^(-8/3) from 1/t_q to infinity [from Wolfram Alpha] (Wiener-Kintchine theorem [2])
        
        phase_psd = 0.033*self.laser.wavenumber**2 * length_step * c2n * windspeed**(5/3) # This is Kolmogorov noise [3].
        return np.sum(phase_psd, axis=1)*freq_integral # This comes from Wiener-Kintchine theorem [2] (units of rad^2).
        
    def generateSim(self, vertical_steps):
        # This generates a 2D array, where with one dimesion being time (even steps), other being height (not even steps)
        actual_heights = self.rec.steps2sat(self.sat, vertical_steps)

        slew_rate = np.array(self.rec.slew2sat(self.sat))[:, np.newaxis] # not sure what units these are desired to be in, not clear from literature, i think rad/s
        windspeed = self.get_windspeed(slew_rate, actual_heights)        # this is the Bufton wind model value of wind speed at each step of height that we are simulating [1]
        rms_windspeed = self.get_wind_rms(actual_heights, windspeed)     # this is the rms of windspeed, used in the calculation of Cn^2 [1]
        c2n = self.get_c2n(rms_windspeed, actual_heights)                # this is our Hufnagel-Valley turbulence model for Cn^2 [1]
        phase_noise_var = self.get_phase_noise_var(c2n, windspeed)       # this is sum of power spectral density of phase noise, with the frequency contribution [2, 3]
        error = phase_noise_var/4                                        # this is QKD QBER from [4]
        return error

seperation = 200e3 # seperation between Alice and Bob (m)

# Hufnagel-Valley turbulence model - [1] Ch. 12
# Note Andrews book is the seminal text
# Vg = 10 # m/s Ground wind speed ([1] suggests 10, 21, 30)
# C2n_0 = 1.7e-14 # Andrews, 2009: Near-ground vertical profile of refractive-index fluctuations (C2n(0)) ([1] suggests 1.7*10^-13 or 1.7*10^-14)


sat_LEO_min = Satellite(200e3, 7.8e3, 60*4) # radius, speed, time of satellite passover
sat_LEO_max = Satellite(1500e3, 7.1e3, 60*18) # radius, speed, time of satellite passover
alice = Receiver(-seperation/(2*earth_radius), V_0=10, C2n_0=1.7e-14)
bob = Receiver(seperation/(2*earth_radius), V_0=10, C2n_0=1.7e-14)
laser = Laser(1550e-9)


def generate_phase_time_plot():
    fig, ax = plt.subplots()#subplot_kw={'projection': 'polar'})
    
    sat_array = [sat_LEO_min, sat_LEO_max] # this is an object of type Satellite
    sat_name = ['LEO min', 'LEO max']
    
    i = 0
    for sat_used in sat_array:
        start = time.time()
        
        max_time = 0.0001
        to_A = Communication(sat_used, alice, laser, max_uninterrupt_time=max_time)
        to_B = Communication(sat_used, bob, laser, max_uninterrupt_time=max_time)
        
        vertical_steps = 100000 # vertical steps have way too big of an effect on the system at the moment, seems like output is proportional to steps
        error_A = to_A.generateSim(vertical_steps) 
        error_B = to_B.generateSim(vertical_steps)

        error = (error_A+error_B)*4 # this is directly from [2] then + 4sin^2(2*pi*f*n*delta_L/c)*ref_laser_noise where delta_L path mismatch abs(AC-BC), n is refractive index
        # would be interesting to know if the error from reference laser is noise from C or when recieved at A and B -> maybe thats what the factor of 4 is for, but i feel like it could also be exponential
        end = time.time()
        print("{} took {}s".format(sat_name[i], end-start))
        ax.plot(sat_used.time_array/sat_used.time, error*100)
        i += 1
    #sites = ['A', 'B']
    #ax.legend(["{} to C, {}km LEO, {} steps".format(site, sat_used.height/1000, steps) for sat_used in sat_array for site in sites])
    ax.legend(["{}km LEO, {} steps".format(sat_used.height/1000, steps) for sat_used in sat_array]) # {} to C,
    ax.set_title("QBER vs Time for integration period {}ms".format(to_A.max_uninterrupt_time *1000)) # Phase Noise @ 1Hz vs Time
    ax.set_xlabel("Normalised Time") # Time (s) / Freq (Hz)
    ax.set_ylabel("QBER (%)") # Phase Noise @ 1 Hz (rad^2)
    ax.set_yscale('log')
    fig.set_size_inches(9, 6)
    
    plt.show()


def generate_visual_plot():
    sat_used = sat_LEO_min
    fig, ax = plt.subplots()
    divisor = 1000
    # Tangent lines r_A_perp, a_A_perp, r_B_perp, a_B_perp = radial_tangent(alice.radius, alice.angle), radial_tangent(bob.radius, bob.angle)
    y, x = TrigFuncs.pol2cart(earth_radius*np.ones(steps)/divisor, np.linspace(0,2*np.pi,steps))
    ax.plot(x, y)
    y, x = TrigFuncs.pol2cart(sat_used.radius/divisor, sat_used.angle)
    ax.plot(x, y, 'bo')
    y, x = TrigFuncs.pol2cart(alice.radius*np.ones(steps)/divisor, alice.angle)
    ax.plot(x, y,'ro')
    y, x = TrigFuncs.pol2cart(bob.radius*np.ones(steps)/divisor, bob.angle)
    ax.plot(x, y,'go')
    ax.set_xlim(-4.5e6/divisor, 4.5e6/divisor)
    ax.set_ylim(5e6/divisor, 8e6/divisor)
    ax.set_title("Model of LEO Satellite Path Around Earth")
    ax.set_xlabel("Distance (km)")
    
    fig.set_size_inches(10, 3.33)
    
    artists = []
    for i in range(200):
        container, = ax.plot([sat_used.y[int(steps/200*i)]/divisor, alice.y/divisor], [sat_used.x[int(steps/200*i)]/divisor, alice.x/divisor], color='red')
        container2, = ax.plot([sat_used.y[int(steps/200*i)]/divisor, bob.y/divisor], [sat_used.x[int(steps/200*i)]/divisor, bob.x/divisor], color='green')
        artists.append([container, container2])
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=30)
    ani.save(filename=r"C:\Users\22503577\OneDrive - UWA\UWA\PhD\7. Code\tmp\pillow_example.gif", writer="pillow")
    plt.show()


#generate_visual_plot()
generate_phase_time_plot()

'''
References:
[1] Laser Beam Propagation through Random Media. Andrews, 2005
[2] Phase Noise in Real-World Twin-Field Quantum Key Distribution. Bertaina, 2023
[3] Optical timing jitter due to atmospheric turbulence: comparison of frequency comb measurements to predictions from micrometeorological sensors. Cladwell, 2020
[4] Coherent phase transfer for real-world twin-ﬁeld quantum key distribution. Clivati, 2022
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
'''

'''
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