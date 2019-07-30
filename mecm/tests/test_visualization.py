from mecm import visualization
import numpy as np
import math
from scipy import interpolate
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import colors, ticker, cm
# from matplotlib import LogNorm
sns.set()

if __name__ == '__main__':

    n_points = 2**7
    fs = 4.0
    t_vect = np.arange(0, n_points) / fs
    f_vect = np.fft.fftfreq(n_points) * fs

    psd_data = np.loadtxt("/Users/qbaghi/Documents/MICROSCOPE/data/session_218/spectre_lisse.txt")

    logpsd_func = interpolate.interp1d(np.log(psd_data[:, 0]), np.log(psd_data[:, 1]), fill_value='extrapolate')

    psd_func = lambda x: np.exp(logpsd_func(np.log(x)))

    cov = visualization.covariance_matrix_time(psd_func, fs, n_points)

    tf_mat = visualization.fourier_transform_matrix(n_points)

    cov_tf = tf_mat.dot(cov.dot(tf_mat.conjugate().transpose()))

    z = np.abs(cov/np.max(cov))
    z_tf = np.abs(cov_tf) / np.max(np.abs(cov_tf))

    # log_norm1 = colors.LogNorm(vmin=z.min(), vmax=z.max())
    # log_norm2 = colors.LogNorm(vmin=z_tf.min(), vmax=z_tf.max())
    #
    # cbar_ticks1 = [math.pow(10, i) for i in
    #                range(math.floor(math.log10(z.min().min())), 1 + math.ceil(math.log10(z.max().max())))]
    # cbar_ticks2 = [math.pow(10, i) for i in
    #                range(math.floor(math.log10(z_tf.min().min())), 1 + math.ceil(math.log10(z_tf.max().max())))]
    #
    # # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig1, ax1 = plt.subplots()
    # sns.heatmap(z, norm=colors.LogNorm(), xticklabels=False, yticklabels=False, cmap="gist_heat", cbar=True, ax=ax1,
    #             cbar_kws={"ticks": cbar_ticks1})
    #
    # ax1.set_xlabel("Time")
    # ax1.set_ylabel("Time")
    #
    # fig2, ax2 = plt.subplots()
    # sns.heatmap(z_tf, norm=colors.LogNorm(), xticklabels=False, yticklabels=False,
    #             cmap="gist_heat",
    #             cbar=True, ax=ax2, cbar_kws={"ticks": cbar_ticks1})
    # ax2.set_xlabel("Frequency")
    # ax2.set_ylabel("Frequency")
    #
    # plt.show()



    # Another way of doing
    y, x = np.meshgrid(np.arange(0, n_points), np.arange(0, n_points)[::-1])
    # Therefore, remove the last value from the z array.
    # z_cut = z[:-1, :-1]
    # z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    z_min, z_max = np.min(np.abs(z)), np.max(np.abs(z))

    fig, ax = plt.subplots()
    # c = ax.imshow(z[], norm=colors.LogNorm())
    c = ax.pcolormesh(x, y, z, cmap='gist_heat', vmin=z_min, vmax=z_max, norm=colors.LogNorm())
    ax.set_title('Time domain')
    # set the limits of the plot to the limits of the data
    # ax.axis([x.min(), x.max(), y.max(), y.min()])
    ax.tick_params(labelleft='off', labelbottom='off')
    fig.colorbar(c, ax=ax)
    plt.show()

    fig4, ax4 = plt.subplots()
    # c = ax.imshow(z[], norm=colors.LogNorm())
    cf = ax4.pcolormesh(x, y, z_tf, cmap='gist_heat', vmin=z_min, vmax=z_max, norm=colors.LogNorm())
    ax4.set_title('Frequency domain')
    # set the limits of the plot to the limits of the data
    # ax4.axis([xf.min(), xf.max(), yf.min(), yf.max()])
    ax4.tick_params(labelleft='off', labelbottom='off')
    fig4.colorbar(cf, ax=ax4)
    plt.show()



