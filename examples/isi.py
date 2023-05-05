"""
Created by Mahmoud Sofy, Fayoum University, Egypt.
LinkedIn: /in/mahmoud-sofy
Email: mahmoudsofy50@gmail.com
Documentation: https://drive.google.com/drive/folders/1JxSbIVKh8lkG3uSuusGFDEhfMqon3k4j?usp=sharing
"""

import serdespy as sdp
import numpy as np

pulse_response = np.load("./data/pulse_response.npy")
pulse_response_ctle = np.load("./data/pulse_response_ctle.npy")
pulse_response_fir_ctle = np.load("./data/pulse_response_fir_ctle.npy")

nyquist_f = 26.56e9
symbol_t = 1/(2*nyquist_f)
samples_per_symbol = 64
t_d = symbol_t/samples_per_symbol

i = sdp.ISI(pulse_response_ctle, t_d, symbol_t)

i.plot_isi_histogram(bins=200)

i.gen_eye(t_d, 200)
i.plot_stat_eye()

i.include_timing()
i.plot_stat_eye(title='Statistical Eye Diagram Including CDR Phase Distribution')

i.plot_ber(50e-3, ber_threshold=1e-24, plot_only=False)
