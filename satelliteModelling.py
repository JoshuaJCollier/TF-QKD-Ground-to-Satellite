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
steps = 100 # no units (fractional time steps) -> the time steps dont seem to have an effect on the system


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

class CartCoordinate():
    def __init__(self, x, y):
        self.x = x
        self.y = y
  
class PolCoordinate():
    def __init__(self, r, a):
        self.r = r
        self.a = a

class Laser():
    def __init__(self, wavelength):
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi/self.wavelength

class Satellite():
    def __init__(self, height, speed, time):
        self.height = height
        self.speed = speed
        self.time = time
        self.time_array = np.linspace(0, self.time, steps)
        self.radius = self.height + earth_radius
        self.angle = self.speed/self.radius * (self.time_array-self.time/2)
        self.x, self.y = pol2cart(self.radius, self.angle) # x and y coordiantes during a timesweep

class Receiver():
    def __init__(self, angle):
        self.angle = angle
        self.radius = earth_radius
        self.x, self.y = pol2cart(self.radius, self.angle)
    
    def dist2sat(self, satellite):
        return magnitude(satellite.x - self.x, satellite.y - self.y)
    
    def steps2sat(self, satellite, hsteps):
        xstep = (satellite.x - self.x) / hsteps
        ystep = (satellite.y - self.y) / hsteps
        mag = magnitude(self.x + xstep[:, np.newaxis] * np.arange(hsteps), self.y + ystep[:, np.newaxis] * np.arange(hsteps)) - earth_radius
        #y, x = pol2cart(r_earth, a_earth)
        #plt.plot(x, y)
        #plt.plot(satellite.x, satellite.y, 'go')
        #plt.plot(xstep * np.arange(hsteps) + self.x, ystep * np.arange(hsteps) + self.y, 'ro')
        #plt.plot(mag)
        #plt.show()
        return mag
    
    def oldsteps2sat(self, satellite, time, hsteps):
        xstep = (satellite.x[time] - self.x) / hsteps
        ystep = (satellite.y[time] - self.y) / hsteps
        mag = magnitude(self.x + xstep * np.arange(hsteps), self.y + ystep * np.arange(hsteps)) - earth_radius
        return mag
    
    def slew2sat(self, satellite):
        ret_arr = np.diff(np.arctan2(satellite.y-self.y, satellite.x-self.x))/(satellite.time/steps)
        return np.concatenate([ret_arr, [ret_arr[-1]]]) # making it the correct length after diff removes an element

class Communication():
    def __init__(self, sat, rec, laser):
        self.sat = sat
        self.rec = rec
        self.laser = laser
        
    def generateSim(self, vertical_steps, Vg, C2n0):
        # This generates a 2D array, where the rows are 
        actual_heights = self.rec.steps2sat(self.sat, vertical_steps)
        height_steps = np.diff(actual_heights, axis=1)
        height_steps = np.concatenate([height_steps, height_steps[:,-1][:, np.newaxis]], axis=1)

        slew_rate = np.array(self.rec.slew2sat(self.sat))[:, np.newaxis]
        wind_speed = slew_rate*actual_heights + Vg + 30*np.exp(-np.power((actual_heights-9400)/4800,2))
        wind_speed_subsection = np.where((actual_heights > 5e3) & (actual_heights < 2e4), wind_speed, 0)
        omega = np.sqrt((1/15e3)*np.trapz(wind_speed_subsection**2, height_steps, axis = 1))
        c2n = 0.00594*np.power(omega[:, np.newaxis]/27, 2) * np.power(1e-5*actual_heights, 10) * np.exp(-actual_heights/1000) + \
                    2.7e-16*np.exp(-actual_heights/1500) + \
                    C2n0*np.exp(-actual_heights/100)
        phase_noise = np.sum(0.033*self.laser.wavenumber**2 * self.rec.dist2sat(self.sat)[:, np.newaxis] * c2n * wind_speed**(5/3), axis=1) * 10**(-8/3)
        
        return phase_noise
#time = 60*5 # time of experimental run (s), 23 minutes is a by eye estimate of usable passover time
#timesteps = np.linspace(0, t, steps) # array of time-steps, used for calculation (array of time in seconds)

seperation = 1000e3 # seperation between Alice and Bob (m)

# Hufnagel-Valley turbulence model - see file:///C:/Users/22503577/Downloads/PM152_ch12.pdf, Andrews, 2005: https://www.spiedigitallibrary.org/ebooks/PM/Laser-Beam-Propagation-through-Random-Media-Second-Edition/eISBN-9780819478320/10.1117/3.626196#_=_
# Note Andrews book is the seminal text
# slew_rate = 0.01 # rate of change of slew from perpective of observer (rad)
Vg = 10 # m/s Ground wind speed (Andrews 2005 suggests 10, 21, 30)
C2n_0 = 1.7e-14 # Andrews, 2009: Near-ground vertical profile of refractive-index fluctuations (C2n(0)) (Andrews 2005 suggests 1.7*10^-13 or 1.7*10^-14)

#LEO_max_arr = {'h':LEO_max, 'r':r_LEO_max, 'a':a_LEO_max}
#LEO_min_arr = {'h':LEO_min, 'r':r_LEO_min, 'a':a_LEO_min}

sat_LEO_min = Satellite(200e3, 7.8e3, 60*4) # radius, speed, time of satellite passover
sat_LEO_max = Satellite(1500e3, 7.1e3, 60*18) # radius, speed, time of satellite passover
alice = Receiver(-seperation/(2*earth_radius))
bob = Receiver(seperation/(2*earth_radius))
laser = Laser(1550e-9)


def radial_tangent(rho, phi):
    tangentSeries = np.arange(-steps/2, steps/2) # array of values to put in to calculate tangent line
    x, y = pol2cart(rho, phi)
    x2 = x + np.cos(phi+np.pi/2)*tangentSeries*1e6
    y2 = y + np.sin(phi+np.pi/2)*tangentSeries*1e6
    r, a = cart2pol(x2, y2)
    return r, a

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

def generate_phase_time_plot2():
    fig, ax = plt.subplots()#subplot_kw={'projection': 'polar'})
    
    sat_array = [sat_LEO_min, sat_LEO_max] # this is an object of type Satellite
    sat_name = ['LEO min', 'LEO max']
    i = 0
    for sat_used in sat_array:
        start = time.time()
        
        to_A = Communication(sat_used, alice, laser)
        to_B = Communication(sat_used, bob, laser)
        
        vertical_steps = 1000 # vertical steps have way too big of an effect on the system at the moment
        phase_A = to_A.generateSim(vertical_steps, Vg, C2n_0) 
        phase_B = to_B.generateSim(vertical_steps, Vg, C2n_0)

        # for 18 mins, integral over freq = 68211
        # for 4 mins, integral over freq = 5561.2
        # more generically integral = 72*2**(1/3)*15**(2/3) / (1/n)**(5/3) where n is minutes
        t_a = sat_used.time
        t_a = 0.0001
        freq_integral = 3 / (5*(1/t_a)**(5/3))
        #freq_integral = 0.012927 # for 100ms transfer window
        error = (phase_A+phase_B)*4*freq_integral # this is directly from Bertaina, 2023 + 4sin^2(2*pi*f*n*delta_L/c)*ref_laser_noise where delta_L path mismatch abs(AC-BC), n is refractive index
        # would be interesting to know if the error from reference laser is noise from C or when recieved at A and B -> maybe thats what the factor of 4 is for, but i feel like it could also be exponential
        end = time.time()
        print("{} took {}s".format(sat_name[i], start-end))
        ax.plot(sat_used.time_array/sat_used.time, error*100)
        #ax.plot(sat_used.time_array, phase_A)
        #ax.plot(sat_used.time_array, phase_B)
        i += 1
    
    #sites = ['A', 'B']
    #ax.legend(["{} to C, {}km LEO, {} steps".format(site, sat_used.height/1000, steps) for sat_used in sat_array for site in sites])
    ax.legend(["{}km LEO, {} steps".format(sat_used.height/1000, steps) for sat_used in sat_array]) # {} to C,
    ax.set_title("QBER vs Time for integration period {}ms".format(t_a*1000)) # Phase Noise @ 1Hz vs Time
    ax.set_xlabel("Normalised Time") # Time (s) / Freq (Hz)
    ax.set_ylabel("QBER (%)") # Phase Noise @ 1 Hz (rad^2)
    #ax.set_yscale('log')
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
    
    


def generate_visual_plot():
    sat_used = sat_LEO_min
    fig, ax = plt.subplots()#subplot_kw = {'projection' : 'polar'})
    divisor = 1000
    # These are radial plots (add subplot_kw={'projection': 'polar'})) that display system
    # Tangent lines
    r_A_perp, a_A_perp = radial_tangent(alice.radius, alice.angle)
    r_B_perp, a_B_perp = radial_tangent(bob.radius, bob.angle)
    
    
    y, x = pol2cart(earth_radius*np.ones(steps)/divisor, np.linspace(0,2*np.pi,steps))
    ax.plot(x, y)
    
    y, x = pol2cart(sat_used.radius/divisor, sat_used.angle)
    ax.plot(x, y)
    #ax.plot(a_A_perp, r_A_perp)
    y, x = pol2cart(alice.radius*np.ones(steps)/divisor, alice.angle)
    ax.plot(x, y,'ro')
    #ax.plot(a_B_perp, r_B_perp)
    y, x = pol2cart(bob.radius*np.ones(steps)/divisor, bob.angle)
    ax.plot(x, y,'go')
    
    #ax.set_rmax(8e6/divisor)
    #ax.set_rlim(6e6/divisor)
    #ax.set_rticks([0, 4e6/divisor, 8e6/divisor])
    #ax.set_xticks(np.pi * np.linspace(-1, 1, 8, endpoint=False))
    #ax.set_thetamin(-90)
    #ax.set_thetamax(90)
    #ax.set_theta_offset(np.pi/2)
    
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
#generate_phase_time_plot()
generate_phase_time_plot2()
#generate_c2n_plot()

'''
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