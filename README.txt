# VirtualPatient

The VirtualPatient Library is composed of the VirtualPatient module and 
a series of examples of its usage applied to ODE models of immune infection.

Virtual Patient is a term used by computational/mathematical modellers in the 
Biomedical field for montecarlo like numerical simulation of biomedical models.

The VirtualPatient.py set of classes allows for simple definition of ODE Models
as well as the data types and methods necessary for storing and simulating ODE 
Models with sampled initial condition and parameter spaces.

Usage is not limited to ODE models in the Biomedical field but any ODE model

## Usage

See VirtualPatient/covid_global.py for an example of model build and simulation
of covid-19 infection.

See VirtualPatient/First_Analysis.py for an example of analyzing the covid
model simulation results using frechet distance along with unsupervised 
learning methods like PCA and Kmeans to understand the effects of infection
on different virtual patients.


## Contributing
This package is not making any further development at this time
