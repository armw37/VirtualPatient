# The Virtual Patient objects
from collections import OrderedDict
import operator as op
from functools import reduce
import numpy as np
import numpy.random as rand
from scipy.integrate import solve_ivp
import h5py

class Samplable_Object():
    """

    Base Class for objects that can be sampled from
    a distribution. Only numpy sampling supported currenty

    Attributes
    ----------
    distribution : str
        A  string of the available distributions that can be sampled from
    sample : array
        a sample from the distribution
    """

    def __init__(self, distribution, sample=None):
        # Test if its in the np.random
        if distribution == None:
            self.distribution = None
        elif distribution in dir(rand):
            self.distribution = eval('rand.' + distribution)
        else:
            raise TypeError('Must choose valid numpy distribution'+ \
                    'or None type')

        if isinstance(sample, np.ndarray):
            self.sample = sample
        elif sample == None:
            self.sample = None
        else:
            raise TypeError('Must choose an array type or None')

    def np_sample(self, args, **kwargs):
        """
        A simple call to the sample function

        Arguments
        ---------

        arg : list
            The list of args to specify the distribution fnction from
            numpy

        returns
        -------
        The sample array or element
        """
        
        self.sample = self.distribution(*args, **kwargs)

    def get_sample(self):
        """
        Get the sample
        """

        return self.sample


class Parameter(Samplable_Object):
    """

    Creates a parameter object to be used in an ODE Model

    Attributes:
    -----------
    name : str : 
        a name for the parameter
    value : float
        a value for the parameter

    """

    def __init__(self, name, value, distribution=None):
        
        if type(name) != str:
            raise TypeError('Name needs to be a string')
        else:
            self.name = name

        if isinstance(value, float) or isinstance(value, int):
            self.value = value
        else:
           raise TypeError('Value needs to be a numeric type')
        
        super(Parameter, self).__init__(distribution)
        

class State_Variable():
    """

    Creates the state variables list

    Attributes
    ----------
    name : str
        Name of the state variable
    initial_condition : Initial_Condition instance
        The initial condition for the variable

    """

    def __init__(self, name, initial_condition):
        if type(name) != str:
            raise TypeError('Name must be a string')
        else:
            self.name = name

        if isinstance(initial_condition, Initial_Condition):
            self.initial_condition = initial_condition
        else:
            raise TypeError('inital_condition must be an' +\
                    ' Initial Condition instance')


class Initial_Condition(Samplable_Object):
    """

    The Initial Conditions of the model

    Attributes:
    -----------
    name : str : 
        a name for the initial condition
    value : float
        a value for the initial condition

    """

    def __init__(self, name, value, distribution=None):

        if type(name) != str:
            raise TypeError('Name needs to be a string')
        else:
            self.name = name

        if isinstance(value, float) or isinstance(value, int):
            self.value = value
        else:
           raise TypeError('Value needs to be a numeric type')
            
        super(Initial_Condition, self).__init__(distribution)

class Equation():
    """

    Create a ODE. Must be first order, Must also be python and 
    C++ evualble.

    Attributes
    ----------
    name : str
        a name for the equation
    equation : str
        the right hand side of the equation using parameter names and
        state_variable names, supports math commands as well
    state_variables : OrderedDict
        An ordered dict of State_Variable instances of the equation
    parameters : OrderedDict
        An Ordered dict of Parameter instances for the equation

    """

    def __init__(self, name, equation):
        if type(name) != str:
            raise TypeError('Name must a string')
        else:
            self.name = name
        
        if type(equation) != str:
            raise TypeError('Name must a string')
        else:
            self.equation = equation

        self.state_variables = OrderedDict()
        self.parameters = OrderedDict()

    def add_parameters(self, params):
        """
        Add parameters as ordered dict

        Arguements
        ----------
        params : list
            A list of parameter objects

        """
        for param in params:
            if isinstance(param, Parameter):
                self.parameters[param.name] = param
            else:
                raise TypeError(str(param) + 
                ' is not a parameter instance')

    def add_state_variable(self, state_vars):
        """
        Add state variables as ordered dict

        Arguements
        ----------
        state_vars : list
            A list of parameter objects

        """
        for state_var in state_vars:
            if isinstance(state_var, State_Variable):
                self.state_variables[state_var.name] = state_var
            else:
                raise TypeError(str(state_var) + 
                        ' is not a State Variable instance')


    def _evaluate(self):
        """

        For Debugging the equations

        returns : RHS derivative
        """

        if self.parameters == None or self.state_variables == None:
            raise TypeError('The equation must have parameters and' +\
                    'state_variables to evaluate')

        for key, value in self.parameters.items():
            exec(key + '=' + str(value.value))

        for key, value in self.state_variables.items():
            exec(key + '=' + str(value.initial_condition.value))

        evaluate = eval(self.equation)

        return evaluate


class Model():
    """
    An ODE Model Base Class

    Attributes
    ----------
    name : str
        The name of the model
    listOfParameters : OrderedDict
        OrderedDict of Parameter instances
    listOfVariables : OrderedDict
        OrderedDict of State_Variable instances
    listOfEquations : OrderedDict
        OrderedDict of Equation instances
    system : Function
        A function of the system of equations 
    """

    def __init__(self, name, sim_time):
        if type(name) == str:
            self.name = name
        else:
            raise TypeError('Name must be a string')

        if isinstance(sim_time, np.ndarray):
            self.sim_time = sim_time
        else:
            raise TypeError('sim_time must be array')

        self.listOfParameters = OrderedDict()
        self.listOfVariables = OrderedDict()
        self.listOfEquations = OrderedDict()

    def add_Parameters(self, param_list):
        """
        Add parameters as ordered dict

        Arguements
        ----------
        param_list : list
            A list of Parameter instances

        """

        for param in param_list:
            if isinstance(param, Parameter):
                self.listOfParameters[param.name] = param
            else:
                raise TypeError(str(param) + 
                ' is not a Parameter instance')
    

    def add_Variable(self, state_vars):
        """
        Add state variables as ordered dict

        Arguements
        ----------
        state_vars : list
            A list of State_Variable instances

        """

        for state_var in state_vars:
            if isinstance(state_var, State_Variable):
                self.listOfVariables[state_var.name] = state_var
            else:
                raise TypeError(str(state_var) + 
                ' is not a State_Variable instance')


    def add_Equation(self, equations):
        """
        Add equations as ordered dict

        Arguements
        ----------
        state_vars : list
            A list of Equation Instances

        """

        for equation in equations:
            if isinstance(equation, Equation):
                self.listOfEquations[equation.name] = equation
            else:
                raise TypeError(str(equation) + 
                        ' is not an Equation instance')


class Patient():
    """
    Creates a Patient, which is a model instance with some identifying
    attributes, and an outcome function which maps the ODE data to some
    target data of interest

    Attributes
    ----------
    ID : str or int
        A unique identifier for each patient
    model : Model instance
        A model instance that defones the patient
    outcome : function
        A function that can be applied to the state variables of
        the ode model and creates categorical or numerical outcome
    """

    def __init__(self, ID, model, sample_dict):
        
        self.model = model
        self.ID = ID
        self.patient_params = sample_dict
        self.build()

    # Build evaluable equations to create unique patient

    def build(self):

        compiled_eq = OrderedDict((eqkey, eqval.equation) for 
                eqkey, eqval in 
                self.model.listOfEquations.items())
        
        def fun(t, y):
            def f(self, t, y):
                curr_state = OrderedDict()
                curr_state['t'] = t
                state_change =  OrderedDict()
                variables = self.model.listOfVariables
                for i, var in enumerate(variables):
                    curr_state[var] = y[i]
            
                for eq_name, eq in compiled_eq.items():
                    state_change[eq_name] = eval(
                        compiled_eq[eq_name], curr_state,
                        self.patient_params)

                state_change = list(state_change.values())
                return state_change
            sol = f(self,t, y)
            return sol

        self.func = fun

    def simulate_patient(self, t_span, t_eval, solver='RK45'):
        # run the model simulation
        func = self.func
        y0 = [var.initial_condition.value for var \
                in self.model.listOfVariables.values()]
        sol = solve_ivp(func, t_span, y0, method=solver, 
            t_eval=t_eval)
        return sol


class Simulation():
    """
    Main class to run a Monte Carlo simulation of the patients 
    """

    def __init__(self, name, length, conditions=None):
        if type(name) == str:
            self.name = name
        else:
            raise TypeError('name must be a string type')
        
        self.listOfPatients = OrderedDict()
        
        if isinstance(length, np.ndarray):
            self.length = length
        else:
            raise TypeError('length must be a linearly spaced\
                    np array')
        
        self.conditions = conditions
    
    def ncr(n, r):
        """
        The combinatorial formula
        """
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

    
    def create_patients(self, model, 
            num_patients, sample_list):

        # calulate # of patients to create
        patients_to_simulate = len(sample_list) * num_patients

        # create unique ID simple list of integers fine
        IDs = [x for x in range(1, patients_to_simulate+1, 1)]

        #create parameter sample lists
        p_dict = {param: \
                (pval.sample if type(pval.sample) != None else \
                pval.value) for param, pval in \
                model.listOfParameters.items()}
        # default dict
        default_dict = {pname : pval.value for pname, pval in \
                model.listOfParameters.items()}

        # Patient Parameter array
        patient_arr = np.array([[pval for pval 
            in default_dict.values()] for i in IDs])

        # slice of paramteres
        slice_dict = {key: i for i, key in 
                enumerate(p_dict.keys())}
        slice_index = [[slice_dict[sample_item] 
            for sample_item in sample] for sample in sample_list]
         
        # Loop samples to create sampled array
        for i, sample in enumerate(sample_list):
            arr = np.array([p_dict[i] for i in sample])
            patient_arr[num_patients*i:num_patients
                    +num_patients*i ,slice_index[i]] = arr.T
        self.patient_arr = patient_arr
        default_patient = np.array([value for value in 
            default_dict.values()])
        self.patient_arr = np.vstack((default_patient, 
            self.patient_arr))
        new_ID = np.array([x for x in range(
            patients_to_simulate + 1)]) 
        self.patient_arr = np.concatenate([np.reshape(new_ID,
            (-1, 1)), self.patient_arr], axis=1)


        # Convert to dictionary
        patient_dict = {x : {key : patient_arr[x-1][i] for 
            i, key in enumerate(p_dict.keys())} for x in IDs}

        # build models
        
        self.listOfPatients[0] = Patient(0, model, default_dict)
        # Build rest of patients
        for key, val in patient_dict.items():
            self.listOfPatients[key] = Patient(key, model, val)

    def run(self, solver='RK45', output_to_file=False, 
            filename=None):
        # run the simulation based off solver and conditions
        # output it based on expected size
        # run default patient to suggest solver
        if output_to_file == False:
            n_patients = len(self.listOfPatients)
            n_rows = self.length.shape[0]
            n_cols = len(self.listOfPatients[0].
                    model.listOfVariables) + 1
            output_arr = np.empty((n_patients,n_rows,n_cols))

        else:
            output_file = h5py.File(filename, 'w')
            
            output_file.create_dataset('meta_data',
                    data=self.patient_arr)
        
        if output_to_file == True and filename == None:
            raise ValueError('If output_to_file is False,' +\
                    'must specify filename')
        
        sol_status =[]
        for key, patient in self.listOfPatients.items():
            t_span = [self.length[0], self.length[-1]]
            sol = patient.simulate_patient(t_span, self.length,
                    solver)
            arr = np.concatenate([np.reshape(sol.t,(-1,1)), 
                        sol.y.T], axis=1)
            
            sol_status.append([key, sol.status])
            if output_to_file == False:
                output_arr[key,:,:] = arr
            else:
                output_file.create_dataset(str(key), data=arr)
        
        if output_to_file == False:
            return output_arr
        else:
            output_file.create_dataset('status', 
                    data=np.array(sol_status))
            output_file.close()

    def import_simulation():
        # import a simulation maybe as a class
        pass

    def export_simulation():
        # some methods of exporting
        pass

    def summary():
        # some kind of printing of summary of the model
        pass

    def analysis_form():
        # put a simmulation into a form that can be analyzed
        pass
