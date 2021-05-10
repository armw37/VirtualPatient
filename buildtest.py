from collections import OrderedDict
from scipy.integrate import solve_ivp
import h5py

compiled_eq = OrderedDict()
compiled_eq['dxdt'] = compile('t1*x - t2*x*y', 'eq', 'eval')
compiled_eq['dydt'] = compile('-t3*y + t4*y*x', 'eq', 'eval')

listOfVariables = ['x', 'y']
sample_dict = {'t1' : 1.5, 't2' : 1, 't3' : 3, 't4': 1}

def f_wrap(t,y):
    #sample_dict = {'t1' : 1.5, 't2' : 1, 't3' : 3, 't4': 1}
    def f(t, y, sample_dict):
        curr_state = OrderedDict()
        curr_state['t'] = t
        state_change =  OrderedDict()
        for i, var in enumerate(listOfVariables):
            curr_state[var] = y[i]

        for eq_name, eq in compiled_eq.items():
            state_change[eq_name] = eval(
                compiled_eq[eq_name], curr_state, sample_dict)

        state_change = list(state_change.values())
        return state_change
    change = f(t,y,sample_dict)
    return change



sol = solve_ivp(f_wrap ,[0, 15], [10, 5])

def lkv(t,z,a,b,c,d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

sol2 = solve_ivp(lkv, [0, 15], [10, 5], args=(1.5, 1, 3, 1))

sol_dict = {'wrapped' : sol.y, 'paramterized' : sol2.y}
with h5py.File('mytestfile.h5py', 'w') as f:
    for key, value in sol_dict.items():
        f.create_dataset(key, data=value)

def lkv2(t,z,a,b,c,d):
    y = z[:2]
    x = z[2:]
    one = lkv(t,y,a,b,c,d)
    two = lkv(t,x,a+1,b+1,c+1,d+1)
    return [one, two]

sol3 = solve_ivp(lkv2, [0, 15], [10, 5, 10,5], 
        args=(1.5, 1, 3, 1), vectorized=True)

print(sol3.y)
