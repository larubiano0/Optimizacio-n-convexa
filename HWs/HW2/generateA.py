import sympy as sp
import sympy.stats as stats

Z = stats.Normal('Z', 0, 1)

n = 5000
m = 2000
A = []

for i in range(m):
    Aj = sp.Matrix(stats.sample(Z, size=n, seed=3527+i))
    norm_Aj = sp.sqrt(sum([a**2 for a in Aj]))
    Aj = (1 / norm_Aj) * Aj # Normalize Aj
    A.append(Aj)
    if i % 100 == 0:
        print(f'{i} vectors generated')

# Save A to a file, where each row is a vector Aj
with open('A.txt', 'w') as f:
    for Aj in A:
        f.write(','.join([str(a) for a in Aj]) + '\n')

