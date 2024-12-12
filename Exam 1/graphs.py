import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os

Fs_for_each = pd.read_csv('Fs_for_each.csv')
# Get first row as list
F10 = Fs_for_each.iloc[0].values.tolist()[1:]
# Get second row as list
F1 = Fs_for_each.iloc[1].values.tolist()[1:]
# Get third row as list
F01 = Fs_for_each.iloc[2].values.tolist()[1:]
# Get fourth row as list
F001 = Fs_for_each.iloc[3].values.tolist()[1:]

Fs_polyak = pd.read_csv('Fs_Polyak.csv')
# Get second column as list
F_polyak = Fs_polyak.iloc[:,1].values.tolist()

Fs_stochastic = pd.read_csv('Fs_stochastic.csv')
# Get fist row as list
F_stochastic10 = Fs_stochastic.iloc[0].values.tolist()[1:]
# Get second row as list
F_stochastic1 = Fs_stochastic.iloc[1].values.tolist()[1:]
# Get third row as list
F_stochastic01 = Fs_stochastic.iloc[2].values.tolist()[1:]
# Get fourth row as list
F_stochastic001 = Fs_stochastic.iloc[3].values.tolist()[1:]

# Graph them all in a single plot, with f_star as a horizontal line

f_star = 0.62988 # Optimal value of the objective function from testing

fig = plt.figure()
# Make x go to the maximum of the lengths of the lists
max_length = max(len(F10), len(F1), len(F01), len(F001), len(F_polyak), len(F_stochastic10), len(F_stochastic1), len(F_stochastic01), len(F_stochastic001))
plt.xlim(0, max_length)
plt.plot(F10, label='R=10', color='blue')
plt.plot(F1, label='R=1', color = 'mediumblue')
plt.plot(F01, label='R=0.1', color = 'darkblue')
plt.plot(F001, label='R=0.01', color = 'navy')
plt.plot(F_polyak, label='Polyak', color = 'black')
plt.plot(F_stochastic10, label='R=10', color = 'darkgreen')
plt.plot(F_stochastic1, label='R=1', color = 'forestgreen')
plt.plot(F_stochastic01, label='R=0.1', color = 'green')
plt.plot(F_stochastic001, label='R=0.01', color= 'seagreen')
plt.axhline(y=f_star, color='r', linestyle='-', label='f_star')
plt.xlabel('Iterations')
plt.ylabel('Objective function value, (all with 5 minutes time limit)')
plt.title('Objective function value vs Iterations')
#Add legend, the first 4 are projected subgradient with stepsize a_k = alpha/sqrt(k+1), for alpha = 10, 1, 0.1, 0.01
# The next one is projected subgradient with Polyak step-size
# The last 4 are stochastic subgradient method with stepsize R = 10, 1, 0.1, 0.01

plt.legend(['Proj. subgrad, alpha = 10', 'Proj. subgrad, alpha = 1', 'Proj. subgrad, alpha = 0.1',
             'Proj. subgrad, alpha = 0.01', 'Polyak', 'Stochastic subgrad, R = 10', 'Stochastic subgrad, R = 1',
               'Stochastic subgrad, R = 0.1', 'Stochastic subgrad, R = 0.01', 'f_star'],
               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})

# Fix so that the legend isn't cut off after saving
fig.tight_layout()


fig.savefig('Objective_function_value_vs_Iterations.png')

plt.clf()

# Graph only the projected subgradient, with f_star as a horizontal line

fig = plt.figure()
plt.xlim(0, max_length)
plt.plot(F10, label='R=10', color='blue')
plt.plot(F1, label='R=1', color = 'green')
plt.plot(F01, label='R=0.1', color = 'orange')
plt.plot(F001, label='R=0.01', color = 'black')
plt.axhline(y=f_star, color='r', linestyle='-', label='f_star')
plt.xlabel('Iterations')
plt.ylabel('Objective function value, (all with 5 minutes time limit)')
plt.title('Objective function value vs Iterations')

plt.legend(['Proj. subgrad, alpha = 10', 'Proj. subgrad, alpha = 1', 'Proj. subgrad, alpha = 0.1',
                'Proj. subgrad, alpha = 0.01', 'f_star'],
                bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})

fig.tight_layout()

fig.savefig('Projected_subgradient_vs_Iterations.png')

plt.clf()

# Graph only the Polyak method, with f_star as a horizontal line

fig = plt.figure()
plt.xlim(0, max_length)
plt.plot(F_polyak, label='Polyak', color = 'black')
plt.axhline(y=f_star, color='r', linestyle='-', label='f_star')
plt.xlabel('Iterations')
plt.ylabel('Objective function value, (all with 5 minutes time limit)')
plt.title('Objective function value vs Iterations')

plt.legend(['Polyak', 'f_star'],
                bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})

fig.tight_layout()

fig.savefig('Polyak_vs_Iterations.png')

plt.clf()

# Graph only the stochastic subgradient method, with f_star as a horizontal line

fig = plt.figure()
plt.xlim(0, max_length)
plt.plot(F_stochastic10, label='R=10', color='blue')
plt.plot(F_stochastic1, label='R=1', color = 'green')
plt.plot(F_stochastic01, label='R=0.1', color = 'orange')
plt.plot(F_stochastic001, label='R=0.01', color = 'black')

plt.axhline(y=f_star, color='r', linestyle='-', label='f_star')

plt.xlabel('Iterations')
plt.ylabel('Objective function value, (all with 5 minutes time limit)')
plt.title('Objective function value vs Iterations')

plt.legend(['Stochastic subgrad, R = 10', 'Stochastic subgrad, R = 1', 'Stochastic subgrad, R = 0.1',
                'Stochastic subgrad, R = 0.01', 'f_star'],
                bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})

fig.tight_layout()

fig.savefig('Stochastic_subgradient_vs_Iterations.png')

plt.clf()

