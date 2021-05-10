# Coupled Cytokine storm model

import VirtualPatient as vp
import matplotlib.pyplot as plt

##########################################################################
# Table 2 and param values
##########################################################################

even_diags = [-5.2, -8.6, -4.4, -8, -3.3, -8.1, -8, -5.5, -8.8]

tab2 = [[-6.413, 0.345, -0.383, -0.186, -0.632, -0.680, -0.206, 0.672,
    -0.818], [-0.554, -18.641, 0.078, 1.576, 1.542, 0.128, 0.184, 0.696,
    -0.903], [-0.487, 0.846, -3.32, 0.145, -0.727, -0.111, -0.030,-0.017, 
    0.617], [0.992, -0.207, 1.566, -13.571, 0.058, -0.823, -0.316, 0.046, 
    -3.356], [0.412, -1.688, -0.303, 0.042, -0.2784, 0.640, 0.769, 0.955,
    0.065], [-1.129, -1.072, -0.278, 0.271, 0.101, -16.305, 0.776, 0.778,
    -0.237], [-0.503, -0.775, 0.422, 0.506, -0.242, -0.022, -15.226, 
    -0.181, -0.957], [0.053, -0.09, -0.376, 0.891, -0.575, 0.227, 0.289,
    -7.571, 0.604], [-0.877, -0.075, 0.275, -0.228, 0.32, 0.343, 1.554,
    -0.271, -19.448]]

cytokine_names = ['TNFp', 'IFNp', 'IL10p', 'IL8p', 'IL6p', 'IL4p', 'IL2p',
        'IL1p','IL12p']

def cyto_iterator(lst, start):
    for i in range(len(lst)):
        yield lst[(i + start) % len(lst)] 

# create the parameter names

param_names = []
for i, name in enumerate(cytokine_names):
    param_names.append([name + '_' + cyto for cyto in cyto_iterator(
        cytokine_names, i)])

rate_p_names = [name + 'r_' + name + 'r' for name in cytokine_names]


##########################################################################
#  parameters
##########################################################################

for i, cyto_eq in enumerate(param_names):
    for j, cyto in enumerate(cyto_eq):
        exec(cyto + '=' + "vp.Parameter('" + cyto + "', " + 
            str(tab2[i][j]) + ", 'uniform')")

for i, rates in enumerate(rate_p_names):
    exec(rates + '=' + "vp.Parameter('" + rates + "', " + 
        str(even_diags[i]) +", 'uniform')")

##########################################################################
#  Initial Conditions
##########################################################################

# concentration
conc_initial = ['TNF_0', 'IFN_0', 'IL10_0', 'IL8_0', 'IL6_0', 'IL4_0', 
    'IL2_0', 'IL1_0','IL12_0']

for conc in conc_initial:
    exec(conc + '=' + "vp.Initial_Condition('" + 
            conc + "', " + str(0) + ')')

# concentraion rate of change
rate_initial = ['TNFr_0', 'IFNr_0', 'IL10r_0', 'IL8r_0', 'IL6r_0', 
    'IL4r_0', 'IL2r_0', 'IL1r_0','IL12r_0']

rate_val = [32831, 55328, 12047, 50804, 16437, 29489, 42780, 35535, 4947]

for i, rate in enumerate(rate_initial):
    exec(rate + '=' + "vp.Initial_Condition('" + rate + "', " + 
        str(rate_val[i]) + ')')

##########################################################################
# State_variables
##########################################################################

conc_vars = ['TNF', 'IFN', 'IL10', 'IL8', 'IL6', 
    'IL4', 'IL2', 'IL1','IL12']

rate_vars = ['TNFr', 'IFNr', 'IL10r', 'IL8r', 'IL6r', 
    'IL4r', 'IL2r', 'IL1r','IL12r']

for i, conc in enumerate(conc_vars):
    exec(conc + '=' + "vp.State_Variable('" + conc + "', " + 
        conc_initial[i] + ')')
    exec(rate_vars[i] + '=' + "vp.State_Variable('" + rate_vars[i] + "', " 
        + rate_initial[i] + ')')

##########################################################################
# Equations
##########################################################################

conc_eqs = ['dTNF', 'dIFN', 'dIL10', 'dIL8', 'dIL6', 
    'dIL4', 'dIL2', 'dIL1','dIL12']

rate_eqs = ['dTNFr', 'dIFNr', 'dIL10r', 'dIL8r', 'dIL6r', 
    'dIL4r', 'dIL2r', 'dIL1r','dIL12r']


obj_list = []
for i, conceq in enumerate(conc_eqs):
    exec(conceq + '=' + "vp.Equation('" + conceq + "', '" 
            + rate_vars[i] + "')")
    obj_list.append(eval(conceq))

for i, obj in enumerate(obj_list):
    obj.add_state_variable([eval(rate_vars[i])])

conc_varobj = [eval(conc) for conc in conc_vars]
for i, rateeq in enumerate(rate_eqs):
    rate_str = rate_p_names[i] + '*' + rate_vars[i]
    conc_str = ''
    for j, conc in enumerate(conc_vars):
        if j == 0:
            conc_str = conc_str + conc + '*' + param_names[i][j]
        else:
            conc_str = conc_str + '+' + conc + '*' + param_names[i][j]

    total_eq = rate_str + '+' + conc_str
    exec(rateeq + '=' + "vp.Equation('" + rateeq + "', '" + total_eq +
            "')")
    eval(rateeq).add_state_variable([eval(rate_vars[i]), *conc_varobj])


##########################################################################
# Model
##########################################################################

sim_time = vp.np.linspace(0, 10, 100)
cyto_storm_model = vp.Model('cyto_storm_model', sim_time)

conc_eqobs = [eval(eq) for eq in conc_eqs] 
rate_eqobs = [eval(eq) for eq in rate_eqs]
cyto_storm_model.add_Equation(conc_eqobs + rate_eqobs)

rate_varobj = [eval(var) for var in rate_vars]
cyto_storm_model.add_Variable(conc_varobj + rate_varobj)

p_list = [eval(p) for sub in param_names for p in sub]
rate_ps_obs = [eval(p) for p in rate_p_names]

cyto_storm_model.add_Parameters(p_list + rate_ps_obs)

##########################################################################
# Run 
##########################################################################

cyto_sim = vp.Simulation('cyto_sim', sim_time)

default_dict = {pname : pval.value for pname, pval in \
                cyto_storm_model.listOfParameters.items()}

cyto_sim.listOfPatients[0] = vp.Patient(0, cyto_storm_model, default_dict)

# simulation test
sim1 = cyto_sim.run(solver='LSODA', output_to_file=False)


##########################################################################
# plotting  
##########################################################################

fig1, axs1 = plt.subplots(3,3, sharex=True, sharey=True, figsize=(10,8))

axs1 = axs1.ravel()
fig1.text(0.5, 0.04, 'Time (days)', ha='center')
fig1.text(0.04, 0.5, 'Concentraion (pg/ml)', va='center', 
        rotation='vertical')
fig1.text(0.5, 0.95, 'Concentration Dynamics of a Cytokine Storm',
        ha='center')

for i, cyto in enumerate(conc_vars):
    axs1[i].plot(sim1[0,:,0], sim1[0,:,i+1], 'k')
    axs1[i].set_title(cyto)

plt.show(block=True)

