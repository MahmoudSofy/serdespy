"""
Created by Mahmoud Sofy, Fayoum University, Egypt.
LinkedIn: /in/mahmoud-sofy
Email: mahmoudsofy50@gmail.com
Documentation: https://drive.google.com/drive/folders/1JxSbIVKh8lkG3uSuusGFDEhfMqon3k4j?usp=sharing
"""

import numpy as np
import scipy.interpolate as spinterp
from scipy.stats import norm
import scipy.special as sp
import matplotlib.pyplot as plt

class ISI:
    
    def __init__(self, pulse_response, dt, ui, main_cursor_sample_no = np.NAN, max_cursors_no = 15):
        """        
        Initialize ISI using pulse response.
        
        Parameters
        ----------
        pulse_response : array
            Pulse response of the channel /w equalization.
        
        dt: float
            Time step used in the pulse response array.
        
        ui: float
            Time of UI.
            
        main_cursor_sample_no: integer
            Number of main cursor sample in the pulse response.
            If no sample number is provided, peak will be used as main cursor sample. 
        
        max_cursors_no: integer
            Maximum number of cursors to consider in ISI calculations.
            Play with it carefully, more cursors require exponentially more memory or very long time (if we did it serially).
        
        """
        
        self.dt = dt
        self.UI = int(round(ui/dt))
        self.pulse_response = pulse_response
        if np.isnan(main_cursor_sample_no):
            self.main_cursor_sample_no = pulse_response.argmax()
        else:
            self.main_cursor_sample_no = main_cursor_sample_no
        self.max_cursors_no = max_cursors_no
        self.isi_vector = self.isi_cursors(self.main_cursor_sample_no)
        self.timing_mask = 1
        self.ber = None
        
        
    def isi_cursors(self, main_cursor):
        """
        This function returns cursors at UI-multiples space from the main cursor.

        Parameters
        ----------
        main_cursor : integer
            Main cursor sample number.

        Returns
        -------
        isi_vector : array
            ISI cursors.
        
        """
        start_point = main_cursor - int(main_cursor/self.UI)*self.UI
        end_point = main_cursor + int((self.pulse_response.size-main_cursor)/self.UI)*self.UI
        isi_vector = np.abs(self.pulse_response[np.arange(start_point, end_point, self.UI)])
        isi_vector.sort()
        isi_vector = isi_vector[-np.arange(1, self.max_cursors_no+1)]
        return isi_vector
        
        
    def unpackbits(self, x, num_bits):
        """
        Similar to numpy.unpackbits but without input size limit.
        Used to vectorize ISI combinations generation.
        This function is posted by Ross, user: 7998652@stackoverflow ...
        as an answer for question 18296035@stackoverflow.
        
        """
        
        if np.issubdtype(x.dtype, np.floating):
            raise ValueError("numpy data type needs to be int-like")
        xshape = list(x.shape)
        x = x.reshape([-1, 1])
        mask = 2**(np.arange(num_bits, dtype=x.dtype)).reshape([1, num_bits])
        return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])
    
    def combinations(self, isi_vector):
        """
        This function generates all possible combinations of ISI.

        Parameters
        ----------
        isi_vector : array
            Array with all ISI cursors.

        Returns
        -------
        isi_combinations : array
            All possible combination values for the given ISI cursors.

        """
        combinations_no = 2**isi_vector.size
        res_sign = self.unpackbits(np.arange(0, combinations_no), isi_vector.size)
        res_sign[res_sign==0] = -1
        isi_combinations = np.matmul(res_sign, isi_vector)
        return isi_combinations
        
    def plot_isi_histogram(self, bins=100, res=150):
        """
        To plot ISI histogram.

        """
        isi_vector = self.isi_cursors(self.main_cursor_sample_no)
        self.isi_combinations = self.combinations(isi_vector)
        plt.figure(dpi=res, layout='constrained')
        plt.hist(self.isi_combinations, bins=bins, density=True)
        plt.title("ISI Distribution")

    def gen_eye(self, time_step, bins):
        """
        Sweeps sampling instant and finds ISI distribution at these points.
        
        Parameters
        ----------
        time_step : float
            Time step used in sweep.

        """
        
        step = int(round(time_step/self.dt))
        intensity = np.zeros([bins, self.UI])
        i = 0
        edge = 1.1*np.max(np.sum(np.abs(self.isi_vector)))
        edges = np.linspace(-edge, edge, bins+1)
        while i<self.UI:
            j = int(self.main_cursor_sample_no - (self.UI/2) + i)
            res_cursors = self.isi_cursors(j)
            hist , junk = np.histogram(self.combinations(res_cursors), bins=edges)
            hist = hist / hist.sum()
            intensity[:,i] = hist
            i = i + step
        self.eye_voltage = 0.5*(edges[1:]+edges[:-1])
        self.eye_time = np.arange(0, self.UI, 1)*self.dt
        self.eye_intensity = intensity
        
    def plot_stat_eye(self, res=150, title='Statistical Eye Diagram'):
        """
        To plot statistical eye distribution.

        """
        
        plt.figure(dpi=res, layout='constrained')
        plt.contourf(self.eye_time, self.eye_voltage, self.eye_intensity * self.timing_mask)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Probability')
        
    def include_timing(self, cdr_phase_dist = np.NAN, cdr_phase_dist_thresh = 0.1):
        """
        This will include CDR phase distribution by ISI distribution.
        
        Parameters
        ----------
        
        cdr_phase_dist: array[2,:] (y; x)
            Distribution of CDR phase noise.
            Assumed to be normal distribution with std = 0.05*UI and nulled below threshold.
            
        cdr_phase_dist_thresh: float
            Ratio below which is set to zero in CDR phase distribution

        """
        
        if np.isnan(cdr_phase_dist):
            x = np.arange(0, self.UI * self.dt, self.dt)
            mu = 0.5 * self.UI * self.dt
            std = 0.05*self.UI*self.dt
            y = norm.pdf(x, mu, std)
            self.cdr_phase_dist = np.array([[y], [x]])
        else:
            self.cdr_phase_dist = cdr_phase_dist
            
        dist = spinterp.interp1d(self.cdr_phase_dist[1,:].flatten(), self.cdr_phase_dist[0,:].flatten())
        timing_dist = dist(self.eye_time)
        timing_dist[timing_dist < cdr_phase_dist_thresh*timing_dist.max()] = 0
        timing_dist = timing_dist / np.sum(timing_dist)
        timing_intensity = np.tile(timing_dist, (np.size(self.eye_voltage), 1))
        self.timing_mask = timing_intensity
        
    def reset_eye(self):
        """
        Undo timing distribution.

        """
        self.timing_mask = 1
        
    def plot_ber(self, sigma, ber_threshold=1e-20, plot_only = False, res=150, title='BER Contours'):
        """
        This function plots BER contours.

        Parameters
        ----------
        sigma : float
            Noise std deviation.
        
        ber_threshold : float
            Value below which ber is set to zero.
            
        plot_only: boolean
            Ckeck it if BER is already calculated and no need to recalculate it.

        """
        if plot_only is False:
            ber = 0.5 * sp.erfc(np.abs(self.eye_voltage)/(np.sqrt(2)*sigma))
            ber_mask = np.tile(ber, (np.size(self.eye_time), 1)).T
            self.ber = self.eye_intensity * self.timing_mask * ber_mask
            self.ber[self.ber<ber_threshold] = np.NAN
            self.ber = np.log10(self.ber)
        plt.figure(dpi=res, layout='constrained')
        plt.contourf(self.eye_time, self.eye_voltage, self.ber)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Log(BER)')