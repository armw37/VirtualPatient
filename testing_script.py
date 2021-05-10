# Script to test virtual patient object file

import VirtualPatient as vp
import numpy as np
from scipy.integrate import solve_ivp

# test parameters
t1 = vp.Parameter('t1', 1.5)
t2 = vp.Parameter('t2', 1,'uniform')
t3 = vp.Parameter('t3', 3, 'normal')
t4 = vp.Parameter('t4', 1)

# test sampling
t2.np_sample([1,2], size=100)
t3.np_sample([3, 0.1, 100])

# probably want to test all types of distributions

# testing Initial_Condition
ic1 = vp.Initial_Condition('ic1', 5)
ic2 = vp.Initial_Condition('ic2', 10, 'normal')
ic2.np_sample([1, 0.5, 100])

# State_Variable
y = vp.State_Variable('y', ic1)
x = vp.State_Variable('x', ic2)

# Equation test
dxdt = vp.Equation('dxdt', 't1*x - t2*x*y')
dydt = vp.Equation('dydt', '-t3*y + t4*y*x')
dxdt.add_parameters([t1, t2])
dxdt.add_state_variable([x, y])
dydt.add_parameters([t3, t4])
dydt.add_state_variable([y, x])
print(dydt._evaluate())
print(dxdt._evaluate())

# Model
sim_time = np.linspace(0, 15, 150)
test_model = vp.Model('test_model', sim_time)
test_model.add_Variable([x, y])
test_model.add_Parameters([t1, t2, t3, t4])
test_model.add_Equation([dxdt, dydt])

# Simulation
test_simulation = vp.Simulation('Test_sim', sim_time)
test_simulation.create_patients(test_model, 100, [['t2'], ['t3']])


def lkv(t,z,a,b,c,d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

sol2 = solve_ivp(lkv, [0, 15], [10, 5], args=(1.5, 1, 3, 1))

# simulation test
sim1 = test_simulation.run()


