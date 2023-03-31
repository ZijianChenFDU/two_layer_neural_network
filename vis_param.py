from save_load import load_network
from matplotlib import pyplot as plt

network = load_network('network.pkl')
params = network.params
W1, W2 = params['W1'], params['W2']

plt.figure(figsize=(6, 6.5))
plt.imshow(W1, cmap='RdBu', interpolation='nearest')
plt.ylabel("Input Layer")
plt.xlabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('W1.png', dpi=128)
plt.savefig('W1.svg')
plt.show()

plt.figure(figsize=(6,6.5))
plt.imshow(W2, cmap='RdBu', interpolation='nearest')
plt.xlabel("Output Layer")
plt.ylabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('W2.png', dpi=128)
plt.savefig('W2.svg')
plt.show()