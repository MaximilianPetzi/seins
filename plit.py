#almost original version
import numpy as np
from matplotlib import pyplot as plt 
nrt=25
am = np.zeros(nrt)
nrs=2
a = np.zeros((nrs,nrt))
for i in range(nrs):
    t = np.load(str(i+1)+'error.npy')
    am += t
    a[i] = t
am = am/nrs
separated = (am[11:2198:3] + am[12:2199:3] + am[13::3] )/3.
mean_error = np.std(a,axis=0)/nrs**2
mean_error_separated = (mean_error[11:2198:3] + mean_error[12:2199:3] + mean_error[13::3] )/3.
markers, caps, bars = plt.errorbar(range(729),separated, yerr=mean_error_separated, label='Direct angle output')
[bar.set_alpha(0.2) for bar in bars]
plt.show()