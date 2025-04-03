
# Python-codes-for-1D-Transient-solution
# Heat Transfer Analysis in a Steel Plate
This repository contains Python and MATLAB codse for analyzing heat transfer and temperature profiles both analytically and numerically using the Finite Difference Method (FDM) with both explicit and implicit schemes. The problem involves a steel plate of thickness 0.02 m initially at a uniform temperature of 200°C. One end of the plate is insulated, while the other is exposed to a cooling medium at a temperature of 50°C with a convective heat transfer coefficient of 100 W/$\text{m}^2$K.

## Problem Description
The heat conduction in the plate follows the one-dimensional unsteady heat conduction equation. The thermal properties of the steel are as follows:
- **Thermal Conductivity (k):** 50 W/m·K
- **Density ($\rho$):** 7800 kg/ $\text{m}^3$
- **Specific Heat Capacity $\text{c}_{\text{p}}$:** 500 J/kg·K

### Governing Equation
The governing equation for the heat conduction in the plate is:
$$T_t - \alpha T_{xx} = 0$$

### Boundary Conditions
- At \(x = 0\): $$T_x (0, t) = 0$$
- At \(x = a\): $$T_x (x, t) - H (T(x, t) - T_0) = 0$$

### Initial Condition
- $$T (x, 0) = T_i$$

where:
- $T_i$ = 200°C
- $T_0$ = 50°C
- $a = 0.02 \text{m}$
- $\alpha = 1.282051 \times 10^{-5} \text{m}^2/\text{s}$
- $k = 50 \text{W/m·K}$
- $H = \frac{h}{k} = 2$
- $h = 100 \text{W/m}^2\text{K}$

## Results
The results include plots of the temperature profiles obtained from the analytical solution and numerical solutions using both explicit and implicit schemes. Additionally, the error between the analytical and numerical solutions is plotted to analyze the accuracy of the numerical methods.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the creators of the Python libraries used in this project, including NumPy, Matplotlib, and SciPy.

## Declaratio of use of AI
This project has utilized AImform for debugging purposes. The AI has been employed to identify and fix bugs in the code, ensuring the functionality and performance of the project. All other code, including the core implementation and features, has been written solely by the editor.
