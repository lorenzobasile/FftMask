import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plt.semilogx([0.01, 0.001, 0.0001, 1e-5, 1e-6], [0.18,0.63,0.77,0.79,0.79], label='Clean')
plt.semilogx([0.01, 0.001, 0.0001, 1e-5, 1e-6], [0.13,0.32,0.41,0.42,0.42], label='Adv')
plt.xticks([0.01, 0.001, 0.0001, 1e-5, 1e-6], [0.01, 0.001, 0.0001, 1e-5, 0])
plt.legend()
plt.xlabel("$\lambda$")
plt.ylabel("Accuracy")
plt.savefig("figures/vgg11/PGD.png")

plt.figure(figsize=(10,8))
plt.semilogx([0.01, 0.001, 0.0001, 1e-5, 1e-6], [0.14,0.63,0.76,0.77,0.77], label='Clean')
plt.semilogx([0.01, 0.001, 0.0001, 1e-5, 1e-6], [0.14,0.32,0.47,0.53,0.54], label='Adv')
plt.xticks([0.01, 0.001, 0.0001, 1e-5, 1e-6], [0.01, 0.001, 0.0001, 1e-5, 0])
plt.legend()
plt.xlabel("$\lambda$")
plt.ylabel("Accuracy")
plt.savefig("figures/vgg11/PGD_INFTY.png")
