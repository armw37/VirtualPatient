# First Test of the Model

import VirtualPatient2 as vp
import numpy as np
from scipy.integrate import solve_ivp
import csv
from collections import Counter
import matplotlib.pyplot as plt

# read in csv matlab results
res_list = []
with open('anita_immune.txt') as res_file:
    csv_reader = csv.reader(res_file, delimiter=',')
    for row in csv_reader:
        res_list.append(row)

matlab_results = np.array(res_list)
matlab_results = np.asfarray(matlab_results, float)

#################################################################
#  parameters
#################################################################

t1 = vp.Parameter('t1', 1.5)
# dV/dt
gammaV    = vp.Parameter('gammaV', 0.5* 510)
gammaVA   = vp.Parameter('gammaVA', 0.5* 619.2)
gammaVH   = vp.Parameter('gammaVH', 0.5* 1.02)
alphaV    = vp.Parameter('alphaV', 0.5* 1.7)
aV1       = vp.Parameter('aV1', 0.5* 100)
aV2       = vp.Parameter('aV2', 23000)
# dH/dt
bHD       = vp.Parameter('bHD', 0.5* 4)
aR        = vp.Parameter('aR', 0.5* 1)
gammaHV   = vp.Parameter('gammaHV', 0.5* 0.34)
bHF       = vp.Parameter('bHF', 0.5* 0.01)
# dI/dt
bIE       = vp.Parameter('bIE', 0.5* 0.066)
aI        = vp.Parameter('aI', 0.5* 1.5)
# dM/dt
bMD       = vp.Parameter('bMD', 0.5* 1, 'uniform')
bMV       = vp.Parameter('bMV', 0.5* 0.0037)
aM        = vp.Parameter('aM', 0.5* 1)
# dF/dt
bF        = vp.Parameter('bF', 0.5* 250000)
cF        = vp.Parameter('cF', 0.5* 2000)
bFH       = vp.Parameter('bFH', 0.5* 17)
aF        = vp.Parameter('aF', 0.5* 8)
# dE/dt
bEM       = vp.Parameter('bEM', 0.5* 8.3, 'uniform')
bEI       = vp.Parameter('bEI', 0.5* 2.72)
aE        = vp.Parameter('aE', 0.5* 0.4)
# dP/dt
bPM       = vp.Parameter('bPM', 0.5* 11.5, 'uniform')
aP        = vp.Parameter('aP', 0.5* 0.4)
# dA/dt
bA        = vp.Parameter('bA', 0.5* 0.043)
gammaAV   = vp.Parameter('gammaAV', 0.5* 146.2)
aA        = vp.Parameter('aA', 0.5* 0.043)
# dS/dt
r         = vp.Parameter('r', 0.5* 3e-5)

# Sample
bMD.np_sample([0.1, 0.5], size=200)
bEM.np_sample([1, 4.15], size=200)
bPM.np_sample([1, 5.75], size=200)

#################################################################
#  Initial Conditions
#################################################################

# testing Initial_Condition
V0 = vp.Initial_Condition('V0', 0.02)
H0 = vp.Initial_Condition('H0', 1)
L0 = vp.Initial_Condition('L0', 0)
I0 = vp.Initial_Condition('I0', 0)
M0 = vp.Initial_Condition('M0', 0)
F0 = vp.Initial_Condition('F0', 0)
R0 = vp.Initial_Condition('R0', 0)
E0 = vp.Initial_Condition('E0', 1)
P0 = vp.Initial_Condition('P0', 1)
A0 = vp.Initial_Condition('A0', 1)
S0 = vp.Initial_Condition('S0', 0.1)

#################################################################
# State_variables
#################################################################

V = vp.State_Variable('V', V0)
H = vp.State_Variable('H', H0)
L = vp.State_Variable('L', L0)
I = vp.State_Variable('I', I0)
M = vp.State_Variable('M', M0)
F = vp.State_Variable('F', F0)
R = vp.State_Variable('R', R0)
E = vp.State_Variable('E', E0)
P = vp.State_Variable('P', P0)
A = vp.State_Variable('A', A0)
S = vp.State_Variable('S', S0)
#D = 1-H-R-I-L;

#################################################################
# Equations
#################################################################

dVdt = vp.Equation('dVdt', 'gammaV*I - gammaVA*S*A*V\
        - gammaVH*H*V - alphaV*V - aV1*V/(1+aV2*V)')
dHdt = vp.Equation('dHdt', 'bHD*(1 -H -R -I -L)*(H+R) + aR*R - \
        gammaHV*V*H - bHF*F*H')
dLdt = vp.Equation('dLdt', 'gammaHV*V*H - 6*L')
dIdt = vp.Equation('dIdt', '6*L - bIE*E*I - aI*I')
dMdt = vp.Equation('dMdt', '(bMD* (1 -H -R -I -L)+ bMV*V)*\
        (1-M) - aM*M')
dFdt = vp.Equation('dFdt', 'bF*M + cF*I - bFH*H*F - aF*F')
dRdt = vp.Equation('dRdt', 'bHF*F*H - aR*R')
dEdt = vp.Equation('dEdt', 'bEM*M*E - bEI*I*E + aE*(1-E)')
dPdt = vp.Equation('dPdt', 'bPM*M*P + aP*(1-P)')
dAdt = vp.Equation('dAdt', 'bA*P - gammaAV*S*A*V - aA*A')
dSdt = vp.Equation('dSdt', 'r*P*(1-S)')

# Add State Variables
dVdt.add_state_variable([I, S, A, V, H])
dHdt.add_state_variable([H, R, I, L, V, F])
dLdt.add_state_variable([V, H, L])
dIdt.add_state_variable([L,E, I])
dMdt.add_state_variable([H, R, I, L, V, M])
dFdt.add_state_variable([M, I, H, F])
dRdt.add_state_variable([F, H, R])
dEdt.add_state_variable([M, E, I])
dPdt.add_state_variable([M, P])
dAdt.add_state_variable([P, S, A, V])
dSdt.add_state_variable([P, S])
# I dont think i need to add params
# dydt.add_parameters([t3, t4])

#################################################################
# Model
#################################################################

sim_time = np.linspace(0, 25, 250)
immune_model = vp.Model('immune_model', sim_time)

immune_model.add_Equation([dVdt, dHdt, dLdt, dIdt, dMdt, dFdt, 
    dRdt, dEdt, dPdt, dAdt, dSdt])

immune_model.add_Variable([V, H, L, I, M , F, R, E, P, A, S])

immune_model.add_Parameters([gammaV, gammaVA, gammaVH, alphaV, 
    aV1, aV2, bHD, aR, gammaHV, bHF, bIE, aI, bMD, bMV, aM, bF,
    cF, bFH, aF, bEM, bEI, aE, bPM, aP, bA, gammaAV, aA, r])
"""
param_values = {name: value.value for name, value in 
        immune_model.listOfParameters.items()}

immune_model.build(param_values)

# test the RHS
t = 0
ics = [var.initial_condition.value for var \
                in immune_model.listOfVariables.values()]

vals = [0.1, .8, .05, 0.05, 0.1, 0.1, 0.1, 1.1, 0.9, 0.9, 0.2] 
rhs = immune_model.func(t, vals)

sol = immune_model.run_as_ODE(solver='LSODA')

# Compare solution
sol_arr = np.concatenate([np.reshape(sol.t,(-1,1)), sol.y.T],
    axis=1)
tol = 0.1
diff = matlab_results - sol_arr
dif_mat = np.divide(diff, matlab_results, out=np.zeros_like(diff),
        where=matlab_results!=0)
truth_mat = np.where(np.absolute(dif_mat) <= tol, 1, 0)
counts = Counter(truth_mat.flatten())
print(counts)
dif_indices = np.argwhere(truth_mat == 0)
rel_dif = dif_mat[dif_indices[:,0], dif_indices[:,1]]
np.savetxt('dif_indices.txt', dif_indices, delimiter=',')
np.savetxt('dif_mat.txt', dif_mat, delimiter=',')
np.savetxt('py_sol.txt', sol_arr, delimiter=',')
np.savetxt('rel_differences.txt', rel_dif, delimiter=',')

# Matching plot
t = sol.t
V = sol_arr[:,1]*1.5e6
H = sol_arr[:,2]
L = sol_arr[:,3]
I = sol_arr[:,4]
M = sol_arr[:,5]
F = sol_arr[:,6]
R = sol_arr[:,7]
E = sol_arr[:,8]
P = sol_arr[:,9]
A = sol_arr[:,10]
S = sol_arr[:,11]
D = 1-H-R-I-L

fig, axs = plt.subplots(4,3)
axs[0, 0].semilogy(t,V)
axs[0, 0].set_ylabel('V');
axs[0, 1].plot(t,H)
axs[0, 1].set_ylabel('H');
axs[0, 2].plot(t,I)
axs[0, 2].set_ylabel('I');
axs[1, 0].plot(t,M)
axs[1, 0].set_ylabel('M');
axs[1, 1].semilogy(t,F)
axs[1, 1].set_ylabel('F');
axs[1, 2].plot(t,R)
axs[1, 2].set_ylabel('R');
axs[2, 0].semilogy(t,E)
axs[2, 0].set_ylabel('E');
axs[2, 1].semilogy(t,P)
axs[2, 1].set_ylabel('P');
axs[2, 2].semilogy(t,A)
axs[2, 2].set_ylabel('A');
axs[3, 0].plot(t,D)
axs[3, 0].set_ylabel('D');
axs[3, 1].plot(t,S)
axs[3, 1].set_ylabel('S');
axs[3, 2].plot(t,L)
axs[3, 2].set_ylabel('L')
fig.tight_layout()
fig.savefig('Default_values.png')
"""
#################################################################
# Simulation
#################################################################

test_simulation = vp.Simulation('Test_sim', sim_time)
test_simulation.create_patients(immune_model, 200, [ 
    ['bEM'], ['bPM'], ['bMD','bEM'], ['bMD','bPM']])

testval = [0.1, 0.8, 0.05, 0.05, 0.1, 0.1, 0.1, 1.1, 0.9, 0.9, 0.2]
t1 = test_simulation.listOfPatients[0].func(0, testval)
t2 = test_simulation.listOfPatients[1].func(0, testval)

# simulation test
sim1 = test_simulation.run(solver='LSODA', output_to_file=True,
        filename='Diabetic_Sim2.h5py')
