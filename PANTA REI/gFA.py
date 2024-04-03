# SimPEG functionality
from SimPEG.utils import plot2Ddata

# Common Python functionality
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tarfile

mpl.rcParams.update({"font.size": 14})

# path to the directory containing our data
dir_path=r"C:\Users\loren\Desktop\\"

# files to work with
coord_filename = dir_path + "coord.txt"
data_filename = dir_path + "gFA.txt"

# Load topography (xyz file)
coord_xyz = np.loadtxt(str(coord_filename))

# Load field data (xyz file)
gFA = np.loadtxt(str(data_filename))


"""Plot Data and Coordinates"""

fig = plt.figure(figsize=(14.9, 10))

ax1 = fig.add_axes([0.08, 0.33, 0.37, 0.62])
plot2Ddata(
    coord_xyz,
    gFA,
    ax=ax1,
    dataloc=True,
    ncontour=40,
    contourOpts={"cmap": "bwr"},
)
ax1.set_title("Free-Air Anomaly", pad=15)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")

# Set axis format to 4 decimals
# ax1.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))
# ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))

cx1 = fig.add_axes([0.09, 0.18, 0.35, 0.04])
norm1 = mpl.colors.Normalize(vmin=-np.max(np.abs(gFA)), vmax=np.max(np.abs(gFA))) #*1000 imposta la scala in mGal
cbar1 = mpl.colorbar.ColorbarBase(
    cx1, norm=norm1, orientation="horizontal", cmap=mpl.cm.bwr
)

cbar1.set_label("$μGal$", size=12)


"""Plot topography"""

ax2 = fig.add_axes([0.56, 0.33, 0.37, 0.62])
plot2Ddata(
    coord_xyz,
    coord_xyz[:, -1],
    ax=ax2,
    dataloc=True,
    ncontour=50,
    contourOpts={"cmap": "gist_earth"},
)
ax2.set_title("Topography", pad=15)
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")

cx2 = fig.add_axes([0.57, 0.18, 0.35, 0.04])
norm2 = mpl.colors.Normalize(vmin=np.min(coord_xyz[:, -1]), vmax=np.max(coord_xyz[:, -1]))
cbar2 = mpl.colorbar.ColorbarBase(
    cx2, norm=norm2, orientation="horizontal", cmap=mpl.cm.gist_earth
)
cbar2.set_label("$m$", size=12)

plt.show()