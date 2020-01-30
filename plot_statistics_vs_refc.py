from matplotlib import pyplot as plt
from read_statistics_vs_refc import read_statistics_vs_refc

models = []
models.append('C13')
models.append('C13+GLM')
models.append('C13+C09')
models.append('C13+C09+GLM')
models.append('C13+C09+C07+GLM')

stats = read_statistics_vs_refc()

fig = plt.figure()

for amodel in models:

    plt.plot(stats['ref'],stats['rmsd'][amodel], label=amodel)

plt.grid()
plt.xlabel('MRMS REFC (dBZ)')
plt.ylabel('RMSD (dBZ)')
plt.legend()

fig.savefig('../OUTPUT/statistics_figures/fig_rmsd_vs_refc.png')
