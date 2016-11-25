font = {'size': 30}
import matplotlib
matplotlib.rc('font', **font)

from matplotlib.ticker import AutoMinorLocator, AutoLocator
from matplotlib.dates import DateFormatter, DateLocator, DayLocator, HourLocator
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers
from utils import load_data
import numpy as np

# need a plot of a full day, then a zoomed in plot

# data = load_data('/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data_3002_3001.csv', 0, use_datetime=True)
data = load_data('/scratch/Dropbox/PhD/htm_models_adelaide/engine/lane_data.csv', 0, use_datetime=True)

day = 288 #timesteps
start_idx = 289 * 49
while data[start_idx][0].hour != 0:
    start_idx += 1
days = 4
x = data[start_idx:start_idx+(days*day) -1, 0]
y = data[start_idx:start_idx+(days*day) -1, 1]
# get all those error values and make them into a new series
ye = np.where(y > 400)[0]
xe = x[ye]
ye = np.random.uniform(-1, 0, ye.shape[0])

y[y>400] = np.nan

fig, ax = plt.subplots()

plt.plot(x, y, 'r-', label='Readings')
# plt.plot(xe, ye, 'bo', label='Errors')
plt.legend(prop={'size':23})
fig.subplots_adjust(bottom=0.125)
majorFormatter = DateFormatter('%a')
ax.xaxis.set_major_formatter(majorFormatter)
majorLocator = DayLocator()

ax.xaxis.set_major_locator(majorLocator)

minorLocator = HourLocator(interval=3)
minorFormatter = DateFormatter('%-I %p')
ax.xaxis.set_minor_formatter(minorFormatter)
ax.xaxis.set_minor_locator(minorLocator)
# # # # for the minor ticks, use no labels; default NullFormatter

# ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
# print (ax.xaxis.get_majorticklabels())
# ax.set_xticks(ax.get_xticks()[::2])
# plt.setp(ax.xaxis.get_ticklabels(which='both'), rotation=90)
# labels = ax.get_xticklabels()
# plt.setp(ax.get_xticks('both'), rotation=30)

for idx,tick in enumerate(ax.xaxis.get_minor_ticks()):
    if idx % 8 == 0:

        tick.label1.set_visible(False)
    tick.label.set_fontsize(26)

    tick.label.set_rotation('vertical')
plt.grid()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='dotted')
# plt.title("Traffic Flow on {} from TS3104 to TS3044".format(x[1].strftime("%A %d %B, %Y")))
plt.title("Traffic Flow from {} to\n {}".format(x[1].strftime("%A %d %B, %Y"), x[-1].strftime("%A %d %B, %Y")), y=1.03)
plt.ylim(0, 300)
plt.ylabel("Total Vehicle Count")
plt.xlabel("Time (h)")
plt.xticks(rotation='vertical')

plt.show()
pass