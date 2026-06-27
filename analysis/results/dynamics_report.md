# Motor Dynamics Analysis

## Fixed-base step response

- Duration: 1 s
- Timestep: 0.001 s
- MuJoCo warnings: 0

| Joint | Target | Final | Error | Peak torque | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| left_roll_joint | 0.1 | -0.0062692 | 0.106269 | 1.18718 | 0.000 |
| left_hip_pitch_joint | 0.1 | -0.0389676 | 0.138968 | 1.56864 | 0.000 |
| left_knee_joint | 0.1 | -0.0101427 | 0.110143 | 0.440786 | 0.000 |
| left_wheel_joint | 0.25 | 0.698594 | -0.448594 | 0.0585881 | 0.000 |
| right_roll_joint | 0.1 | 0.0343791 | 0.0656209 | 1.09881 | 0.000 |
| right_hip_pitch_joint | 0.1 | 0.17162 | -0.0716201 | 1.12333 | 0.000 |
| right_knee_joint | 0.1 | 0.0137395 | 0.0862605 | 0.32844 | 0.000 |
| right_wheel_joint | 0.25 | -0.492835 | 0.742835 | 0.0950015 | 0.000 |

## Free-base posture check

- Duration: 1 s
- Steps: 1000
- MuJoCo warnings: 0
- Peak |qvel|: 9.32212
- Peak |ctrl|: 11.9524
- Final base height: 0.139847 m
- Interpretation: Balance control is not implemented; free-base falling is allowed when the simulation remains finite and MuJoCo reports no warnings.
