from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from utils import load_data
# need a plot of a full day, then a zoomed in plot

data = load_data('/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data.csv', 0, use_datetime=True)

day = 288 #timesteps
start_idx = 129314
days = 4
x = data[start_idx:start_idx+(days*day), 0]
y = data[start_idx:start_idx+(days*day), 1]
fig, ax = plt.subplots()
plt.plot(x, y, 'r-', label='Readings')
plt.legend()

majorFormatter = DateFormatter('%A')
minorLocator = AutoMinorLocator(n=6)
minorFormatter = DateFormatter('%-I %p')
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_formatter(minorFormatter)
# # for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)

ax.set_xticks(ax.get_xticks()[::2])
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='dotted')
plt.title("Traffic Flow from {} to\n {}".format(x[1].strftime("%A %d %B, %Y"), x[-1].strftime("%A %d %B, %Y")))
plt.ylabel("Vehicles/ 5 min")
plt.xlabel("Time")
plt.xticks(rotation='vertical')
plt.show()