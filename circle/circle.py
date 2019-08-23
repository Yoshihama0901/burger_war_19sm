#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
df.columns = ['enemy_x', 'enemy_y', 'enemy_qx', 'enemy_qy', 'enemy_qz', 'enemy_qw', 'enemy_ax', 'enemy_ay', 'enemy_az', 'my_x', 'my_y', 'my_qx', 'my_qy', 'my_qz', 'my_qw', 'my_ax', 'my_ay', 'my_az', 'circle_x', 'circle_y', 'circle_r'] # Fix my and enemy

# Fix to convert gazebo coordinate to amcl_pose coordinate
df['my_x0'] = df['my_x']
df['enemy_x0'] = df['enemy_x']
df['my_x'] = df['my_y']
df['enemy_x'] = df['enemy_y']
df['my_y'] = -df['my_x0']
df['enemy_y'] = -df['enemy_x0']
df['my_az'] = df['my_az'] - 90

df = df[['my_x', 'my_y', 'my_az', 'enemy_x', 'enemy_y', 'circle_x', 'circle_y', 'circle_r']]
df = df[(df['circle_x'] >= 0) & (df['circle_y'] < 300)]
df['dx'] = df['enemy_x'] - df['my_x']
df['dy'] = df['enemy_y'] - df['my_y']
df['u'] = np.cos(np.deg2rad(-df['my_az'])) * (-df['dy']) - np.sin(np.deg2rad(-df['my_az'])) * df['dx']
df['v'] = np.sin(np.deg2rad(-df['my_az'])) * (-df['dy']) + np.cos(np.deg2rad(-df['my_az'])) * df['dx']
df = df.drop(['dx', 'dy'], axis=1)
df['theta'] = np.rad2deg(np.arctan2(df['v'], df['u']) - np.pi / 2)
df['sin_theta'] = np.sin(np.deg2rad(df['theta']))
df.to_csv(sys.stdout, float_format='%.6f')
