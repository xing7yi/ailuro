### Problem description

Compare results of contact simulation between the ANSYS and MOOSE.


**Model Type**
- Transient Mechanics
- 2D Axisymmetric model


**Geometry** 
- Sphere (1/4 circle)
  - diameter: 27.485 um
- Indenter (rigid flat)

**Material**

  - Friction coefficient: 0.1

- Elastic: 
  - Young's modulus (E): 193 GPa
  - Poisson's ratio ($\nu$): 0.3

- Plastic:
  - Yield strength ($\sigma_0$): 400 MPa
  - Hardening constant ($R_0$): 800 MPa
  - Exponential coefficient ($R_\infty$): 300 MPa
  - Exponential saturation parameter ($b$): 10

**Solve Condition**
- Maximum Force: 752 mN
- Compression ratio: 0.48
- Time(s): 11.55 s





