from prepare_data import prepare_data
import os
import matplotlib.pyplot as plt
import numpy as np

# why is estimated better ? huh?

path = os.path.join(os.getcwd(), "test-data", "07-29-default", "IMMEC_0ecc_5.0sec.npz")
# path = os.path.join(os.getcwd(), "test-data","07-29-nonlin-nonzero-initload","IMMEC_nonlin_0ecc_5.0sec.npz")
#path = os.path.join(os.getcwd(), "test-data", "07-31-50ecc-load", "IMMEC_50ecc_5.0sec.npz")

data_real = prepare_data(path, test_data=True, use_estimate_for_v=False)
data_esti = prepare_data(path, test_data=True, use_estimate_for_v=True)

v_real = data_real["u"][:, :3]
v_esti = data_esti["u"][:, :3]

fig, ax = plt.subplots()
l1, _,_ = ax.plot(data_real['t'][:,:,0],v_real, 'k--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [V]')
l2, _,_ = ax.plot(data_esti['t'][:,:,0],v_esti, ':')
ax.legend([l1,l2],["Exact calculation", "Approx from line voltages"])
ax.set_xlim(3.15, 3.40)

left, bottom, width, height = [0.2, 0.6, 0.25, 0.25]
ax_inset = fig.add_axes([left, bottom, width, height])
# Plot the zoomed-in data
ax_inset.plot(data_real['t'][:,:,0],v_real, 'k--')
ax_inset.plot(data_esti['t'][:,:,0],v_esti, ':')
ax_inset.set_xlim(3.25, 3.30)  # adjust limits of the inset
ax_inset.set_ylim(-1.2, 1.2)

# make fourier spectrum of both
# Simulated current signal
dt = 5e-5
n_fft = 5.0 / dt
sampling_freq = 1 / dt


# Perform FFT on zero'th component
print("Supply frequency: ", 50/400*np.max(v_real))


fft_vreal = np.fft.fft(v_real[:,-1])
fft_vesti = np.fft.fft(v_esti[:,-1])

pvreal = np.abs(fft_vreal / n_fft)[:int(n_fft / 2 + 1)]
pvesti = np.abs(fft_vesti / n_fft)[:int(n_fft / 2 + 1)]

pvreal[1:-1] = 2 * pvreal[1:-1]
pvesti[1:-1] = 2 * pvesti[1:-1]

freq = sampling_freq * np.arange(0, pvesti.shape[0]) / n_fft
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(freq, pvreal, label="FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Single-Sided FFT of V_0 real")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(freq, pvesti, label="FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Single-Sided FFT of V_0 estimated")
plt.grid()

plt.tight_layout()
plt.show()



#v_diff = np.abs(v_real - v_esti)
#plt.figure()
#plt.semilogy(v_diff)
plt.show()
