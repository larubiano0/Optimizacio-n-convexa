import matplotlib.pyplot as plt

X = [i*0.01 for i in range(10000)]
Y = [i*0.01 for i in range(10000)]

alpha = [2,3]
a = 0.5
b = 3

final_set = []
for x in X:
    for y in Y:
        if a<=alpha[0]*x + alpha[1]*y<=b:
            final_set.append((x,y))

# Plot final set

plt.scatter([x[0] for x in final_set], [x[1] for x in final_set])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Region factible')
plt.show()
