Model developed for solving the three dimensional heat transfer equation and gas diffusion equation via a finite volume scheme using the Douglas 1962 ADI method for application to granular media and comparison with laboratory measurements.

Currently implemented features:
  - Solving the three dimensional heat transfer and gas diffusion equation concurrently
  - Sintering and temperature dependent calculation of thermal parameters for water ice and other granular materials
  - Rudimentary ray tracing to support reradiated heat and scattered light energy transfer

The model ist designed to be modular with each script containing different functions specified by the name of the script.
The base code is formed by the modules "heat_transfer_equation_ADI_DG.py" and "tri_diag_solve.py" which contain functions needed to solve the heat transfer equation numerically.
Functions from the other modules can be used to create a specific model to the problem at hand.

"three_d_hte_ices.py" contains examples for 2 codes.

Model created for the Master's Thesis "Thermophysikalische Modellierung granularer Medien im Vakuum/Thermophysical modelling of granular media in vacuum conditions" Schuckart 2024.
