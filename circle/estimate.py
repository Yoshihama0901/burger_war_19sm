#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])

df['est_enemy_sin_theta'] = -0.00143584 * df['circle_x'] \
                    + 0.4458366274811388
df['est_enemy_theta'] = np.rad2deg(np.arcsin(df['est_enemy_sin_theta']))
df['est_enemy_v'] = 4.58779425e-09 * np.power(df['circle_y'], 4) \
                    - 1.14983273e-06 * np.power(df['circle_y'], 3) \
                    + 1.21335973e-04 * np.power(df['circle_y'], 2) \
                    - 7.94065667e-04 * df['circle_y'] \
                    + 0.5704722921109504
df['est_enemy_u'] = df['est_enemy_v'] * np.tan(np.deg2rad(df['est_enemy_theta']))
df['est_dx'] = np.cos(np.pi / 2 - np.deg2rad(df['my_az'])) * df['est_enemy_u'] \
               + np.sin(np.pi / 2 - np.deg2rad(df['my_az'])) * df['est_enemy_v']
df['est_dy'] = - np.sin(np.pi / 2 - np.deg2rad(df['my_az'])) * df['est_enemy_u'] \
               + np.cos(np.pi / 2 - np.deg2rad(df['my_az'])) * df['est_enemy_v']
df['est_enemy_x'] = df['my_x'] + df['est_dx']
df['est_enemy_y'] = df['my_y'] + df['est_dy']

df['diff_enemy_u'] = np.abs(df['est_enemy_u'] - df['u'])
df['diff_enemy_v'] = np.abs(df['est_enemy_v'] - df['v'])
df['diff_enemy_theta'] = np.abs(df['est_enemy_theta'] - df['theta'])
df['diff_enemy_x'] = np.abs(df['est_enemy_x'] - df['enemy_x'])
df['diff_enemy_y'] = np.abs(df['est_enemy_y'] - df['enemy_y'])
df['diff_enemy_sin_theta'] = np.abs(df['est_enemy_sin_theta'] - df['sin_theta'])

df.to_csv(sys.stdout, float_format='%.6f')
