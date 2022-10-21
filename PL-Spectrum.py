import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', size=12)  # controls default text sizes
plt.rc('axes', titlesize=42)  # fontsize of the axes title
plt.rc('axes', labelsize=42)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=34)  # fontsize of the tick labels
plt.rc('ytick', labelsize=34)  # fontsize of the tick labels
plt.rc('figure', titlesize=8)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = 2

file = 'spectrum3.txt'
df = pd.read_csv(file, sep='\t')

lam = df.iloc[:, 0]
intensity = df.iloc[:, 1]

fig, ax = plt.subplots(figsize=(18, 12))  # , constrained_layout=True)
plt.plot(lam, intensity / max(intensity), 'b-')
plt.xlabel('Wavelength [nm]', labelpad=8)
plt.ylabel('Normalized Intensity [a.u.]', labelpad=15)
plt.xticks(y=-0.01)
plt.yticks(x=-0.005)
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='grey', linestyle='-', alpha=0.1)
plt.xlim(450, 950)
# plt.ylim(0.05, 0.35)
plt.tick_params(width=2.5, length=10, top=True, right=True, direction='in')
plt.title(file, pad=55)
plt.show()
