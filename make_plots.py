from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from utils import load_data
import numpy as np
# need a plot of a full day, then a zoomed in plot

data = load_data('/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data_3002_3001.csv', 0, use_datetime=True)

day = 288 #timesteps
start_idx = 139000
days = 40
x = data[start_idx:start_idx+(days*day), 0]
y = data[start_idx:start_idx+(days*day), 1]
# get all those error values and make them into a new series
ye = np.where(y>400)[0]
xe = x[ye]
ye = np.random.uniform(-1,0, ye.shape[0])


y[y>400] = np.nan

fig, ax = plt.subplots()
plt.plot(x, y, 'r-', label='Readings')
plt.plot(xe, ye, 'bo', label='Errors')
plt.legend()

majorFormatter = DateFormatter('%A')
minorLocator = AutoMinorLocator(n=6)
minorFormatter = DateFormatter('%-I %p')
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_formatter(minorFormatter)
# # for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)

ax.set_xticks(ax.get_xticks()[::2])
plt.grid()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='dotted')
plt.title("Traffic Flow from {} to\n {}".format(x[1].strftime("%A %d %B, %Y"), x[-1].strftime("%A %d %B, %Y")))
plt.ylim(-1, 120)
plt.ylabel("Vehicles/ 5 min")
plt.xlabel("Time")
plt.xticks(rotation='vertical')
plt.show()
