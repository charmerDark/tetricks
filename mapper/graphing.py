import matplotlib.pyplot as plt
import numpy as np

# Define the range of n values
n = np.arange(1, 30)  # n from 1 to 10

# Calculate the values for each mapper
cgra_mapper = 44 + (n - 1) * 10
our_mapper = 9 + (n - 1) * 9

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(n, cgra_mapper, marker='o', label='CGRA-Mapper')
plt.plot(n, our_mapper, marker='s', label='Our mapper')

plt.xlabel('n')
plt.ylabel('Value')
plt.title('Mapper comparision einsum("ij,jk,kl.lm,mn->in")')
plt.legend()
plt.grid(True)
plt.show()
