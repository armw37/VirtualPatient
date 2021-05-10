# First Test of the Model

import VirtualPatient as vp
import numpy as np
from scipy.integrate import solve_ivp
import csv
from collections import Counter
import matplotlib.pyplot as plt

#################################################################
#  parameters
#################################################################

t1 = vp.Parameter('t1', 1.5)
# dV/dt
gammaV    = vp.Parameter('gammaV', 0.5* 510, 'uniform')
gammaVA   = vp.Parameter('gammaVA', 0.5* 619.2, 'uniform')
gammaVH   = vp.Parameter('gammaVH', 0.5* 1.02, 'uniform')
alphaV    = vp.Parameter('alphaV', 0.5* 1.7, 'uniform')
aV1       = vp.Parameter('aV1', 0.5* 100, 'uniform')
aV2       = vp.Parameter('aV2', 23000, 'uniform')
# dH/dt
bHD       = vp.Parameter('bHD', 0.5* 4, 'uniform')
aR        = vp.Parameter('aR', 0.5* 1, 'uniform')
gammaHV   = vp.Parameter('gammaHV', 0.5* 0.34, 'uniform')
bHF       = vp.Parameter('bHF', 0.5* 0.01, 'uniform')
# dI/dt
bIE       = vp.Parameter('bIE', 0.5* 0.066, 'uniform')
aI        = vp.Parameter('aI', 0.5* 1.5, 'uniform')
# dM/dt
bMD       = vp.Parameter('bMD', 0.5* 1)
bMV       = vp.Parameter('bMV', 0.5* 0.0037, 'uniform')
aM        = vp.Parameter('aM', 0.5* 1, 'uniform')
# dF/dt
bF        = vp.Parameter('bF', 0.5* 250000, 'uniform')
cF        = vp.Parameter('cF', 0.5* 2000, 'uniform')
bFH       = vp.Parameter('bFH', 0.5* 17, 'uniform')
aF        = vp.Parameter('aF', 0.5* 8, 'uniform')
# dE/dt
bEM       = vp.Parameter('bEM', 0.5* 8.3)
bEI       = vp.Parameter('bEI', 0.5* 2.72, 'uniform')
aE        = vp.Parameter('aE', 0.5* 0.4)
# dP/dt
bPM       = vp.Parameter('bPM', 0.5* 11.5)
aP        = vp.Parameter('aP', 0.5* 0.4)
# dA/dt
bA        = vp.Parameter('bA', 0.5* 0.043, 'uniform')
gammaAV   = vp.Parameter('gammaAV', 0.5* 146.2, 'uniform')
aA        = vp.Parameter('aA', 0.5* 0.043, 'uniform')
# dS/dt
r         = vp.Parameter('r', 0.5* 3e-5, 'uniform')

# Sample

# dV/dt
gammaV.np_sample([(0.5* 510)-(0.5*0.5*510),
    (0.5* 510)+(0.5*0.5*510)], size=100)

gammaVA.np_sample([(0.5* 619.2)-(0.5*0.5* 619.2),
    (0.5* 619.2)+(0.5*0.5* 619.2)], size=100)

gammaVH.np_sample([(0.5* 1.02)-(0.5*0.5* 1.02),
    (0.5* 1.02)+(0.5*0.5* 1.02)], size=100)

alphaV.np_sample([(0.5* 1.7)-(0.5*0.5* 1.7),
    (0.5* 1.7)+(0.5*0.5* 1.7)], size=100)

aV1.np_sample([(0.5* 100)-(0.5*0.5* 100),
    (0.5* 100)+(0.5*0.5* 100)], size=100)

aV2.np_sample([23000-23000*0.5, 23000+23000*0.5], size=100)

# dH/dt
bHD.np_sample([(0.5* 4)-(0.5*0.5*4),
    (0.5* 4)+(0.5*0.5*4)], size=100)

aR.np_sample([0.25, 0.75], size=100)

gammaHV.np_sample([(0.5* 0.34)-(0.5*0.5* 0.34),
    (0.5* 0.34)+(0.5*0.5* 0.34)], size=100)

bHF.np_sample([(0.5* 0.01)-(0.5*0.5* 0.01),
    (0.5* 0.01)+(0.5*0.5* 0.01)], size=100)

# dI/dt
bIE.np_sample([(0.5* 0.066)-(0.5*0.5* 0.066),
    (0.5* 0.066)+(0.5*0.5* 0.066)], size=100)

aI.np_sample([0.75, 2.25], size=100)

# dM/dt
bMV.np_sample([(0.5* 0.0037)-(0.5*0.5* 0.0037),
    (0.5* 0.0037)+(0.5*0.5* 0.0037)], size=100)

aM.np_sample([0.25, 0.75], size=100)
# dF/dt
bF.np_sample([(0.5* 250000)-(0.5*0.5* 250000),
    (0.5* 250000)+(0.5*0.5* 250000)], size=100)

cF.np_sample([500,1500], size=100)

bFH.np_sample([(0.5* 17)-(0.5*0.5* 17),
    (0.5* 17)+(0.5*0.5* 17)], size=100)

aF.np_sample([2,6], size=100)

# dE/dt
bEI.np_sample([(0.5* 2.72)-(0.5*0.5* 2.72),
    (0.5* 2.72)+(0.5*0.5* 2.72)], size=100)

# dA/dt
bA.np_sample([(0.5* 0.043)-(0.5*0.5* 0.043),
    (0.5* 0.043)+(0.5*0.5* 0.043)], size=100)
    
gammaAV.np_sample([(0.5* 146.2)-(0.5*0.5* 146.2),
    (0.5* 146.2)+(0.5*0.5* 146.2)], size=100)

aA.np_sample([(0.5* 0.043)-(0.5*0.5* 0.043),
    (0.5* 0.043)+(0.5*0.5* 0.043)], size=100)

# dS/dt
r.np_sample([(0.5* 3e-5)-(0.05*0.5* 3e-5),
    (0.5* 3e-5)+(0.05*0.5* 3e-5)], size=100)

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

#################################################################
# Simulation
#################################################################
sample_list= [['gammaV'],['gammaVA'],['gammaVH'],['alphaV'],
        ['aV1'],['aV2'],['bHD'],['aR'],['gammaHV'],['bHF'],
        ['bIE'],['aI'],['bMV'],['aM'],['bF'],['cF'],['bFH'],
        ['aF'],['bEI'],['bA'],['gammaAV'],['aA'],['r']]

test_simulation = vp.Simulation('Test_sim', sim_time)
test_simulation.create_patients(immune_model, 100, sample_list)

testval = [0.1, 0.8, 0.05, 0.05, 0.1, 0.1, 0.1, 1.1, 0.9, 0.9, 0.2]
t1 = test_simulation.listOfPatients[0].func(0, testval)
t2 = test_simulation.listOfPatients[1].func(0, testval)

# simulation test
sim1 = test_simulation.run(solver='LSODA', output_to_file=True,
        filename='Diabetic_Sim3.h5py')
