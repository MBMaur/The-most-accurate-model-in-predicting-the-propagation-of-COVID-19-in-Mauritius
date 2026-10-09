import numpy as np
K_fixed = 330415
r = 0.0178
P0_fixed = 4372

for t in range (0,1336):
   Y = (P0_fixed * K_fixed * np.exp(r * t)) / ((K_fixed - P0_fixed) + P0_fixed * np.exp(r * t))
   B = Y
   print(Y)

