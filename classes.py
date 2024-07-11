import numpy as np

earth_radius = 6371e3 # radius of earth (m)
speed_of_light = 2.99792458*1e8 # (m/s)

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
        steps = len(rho) # this might not work lmao but doesnt matter I dont use it anymore
        tangentSeries = np.arange(-steps/2, steps/2) # array of values to put in to calculate tangent line
        x, y = TrigFuncs.pol2cart(rho, phi)
        x2 = x + np.cos(phi+np.pi/2)*tangentSeries*1e6
        y2 = y + np.sin(phi+np.pi/2)*tangentSeries*1e6
        r, a = TrigFuncs.cart2pol(x2, y2)
        return r, a

class Laser():
    def __init__(self, wavelength: float):
        self.wavelength = wavelength # wavelength (m)
        self.wavenumber = 2*np.pi/self.wavelength # angular wave number (m^-1)
        self.frequency = speed_of_light/self.wavelength # freq of light (Hz)

class Satellite():
    def __init__(self, height: float, ground_sep: float, max_angle: float, no_time_steps):
        self.height = height
        #self.speed = speed
        #self.time = time
        offset_sep_angle = ground_sep/(2*earth_radius) # angle in radians
        rec_to_sat_angle = max_angle - np.arcsin(earth_radius/(earth_radius+height)*np.sin(np.pi-max_angle))
        total_arc = 2*(rec_to_sat_angle - offset_sep_angle)
        self.speed_m = np.sqrt(6.673e-11*5.98e24/(earth_radius+height)) # sqrt(G*M_E/(R_sat)) in meters/second
        self.speed = self.speed_m/((earth_radius+height)) # speed in rad/s 
        self.time = total_arc/self.speed
        self.no_time_steps = no_time_steps
        self.time_array = np.linspace(0, self.time, no_time_steps)
        self.radius = self.height + earth_radius
        self.angle = self.speed * (self.time_array-self.time/2)
        #self.speed/self.radius * (self.time_array-self.time/2)
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
    
    def steps2sat(self, satellite: Satellite, no_steps):
        
        xstep = (satellite.x[:, np.newaxis] - self.x) * np.arange(no_steps) / no_steps
        ystep = (satellite.y[:, np.newaxis] - self.y) * np.arange(no_steps) / no_steps
        return TrigFuncs.magnitude(self.x + xstep, self.y + ystep) - earth_radius
    
    def lengthsteps2sat(self, satellite: Satellite, no_steps):
        xstep = (satellite.x[:, np.newaxis] - self.x) / no_steps * np.arange(no_steps+1)
        ystep = (satellite.y[:, np.newaxis] - self.y) / no_steps * np.arange(no_steps+1)
        
        return TrigFuncs.magnitude(np.diff(xstep), np.diff(ystep))

    
    def slew2sat(self, satellite: Satellite):
        """ Slew rate of the satellite from the perspective of the receiver.

        Args:
            satellite (Satellite): The satellite that is passing overhead.

        Returns:
            slew_rate (arr): 1D array of slew rate of satellite at each time step (in radians/second).
        """
        ret_arr = np.diff(np.arctan2(satellite.y-self.y, satellite.x-self.x))/(satellite.time/satellite.no_time_steps)
        return np.concatenate([ret_arr, [ret_arr[-1]]]) # arc tan will return the value in radians per second
