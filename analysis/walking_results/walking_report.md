# Walking Control Analysis

Walking v1 keeps the standing leg pose and uses wheel-speed control.
The current convention is: positive forward velocity maps to world -X.

- Target forward velocity: 0.25 m/s
- Ramp time: 2 s
- Duration: 8 s
- Velocity averaging window: 2 s
- MuJoCo warnings: 0
- Finite state: True
- Walking objective met: True
- Walking score: 0.305965
- Forward distance: 2.19118 m
- World X displacement: -2.19118 m
- Final forward velocity: 0.253799 m/s
- Average forward velocity last window: 0.250946 m/s
- Average forward velocity error: 0.000945639 m/s
- Peak |pitch|: 0.100412 rad
- Final pitch: 0.0832534 rad
- Peak |pitch rate|: 0.167928 rad/s
- Peak |wheel torque|: 1.0675 N·m
- Wheel torque saturation fraction: 0.000
- Non-wheel ground contact count: 0
- Non-wheel ground contact geoms: none
- Final base height: 0.28219 m

This is wheel-speed walking v1, not legged gait generation or turning.
