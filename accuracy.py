import matplotlib.pyplot as plt

plt.figure()
plt.semilogx([0.01, 0.001, 0.0001, 1e-5], [0.15,0.63,0.75,0.76])
plt.semilogx([0.01, 0.001, 0.0001, 1e-5], [0.15,0.32,0.46,0.53])
#plt.semilogx([0.01, 0.001, 0.0001, 1e-5], [0.19,0.64,0.76,0.8])
#plt.semilogx([0.01, 0.001, 0.0001, 1e-5], [0.19,0.58,0.66,0.65])
#plt.hlines([0.86, 0.23])
plt.savefig("figures/vgg11/PGD.png")
