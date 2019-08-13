#!/usr/bin/env python
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
df.columns = ['enemy_x', 'enemy_y', 'enemy_qx', 'enemy_qy', 'enemy_qz', 'enemy_qw', 'enemy_ax', 'enemy_ay', 'enemy_az', 'my_x', 'my_y', 'my_qx', 'my_qy', 'my_qz', 'my_qw', 'my_ax', 'my_ay', 'my_az', 'circle_x', 'circle_y', 'circle_r'] # Fix my and enemy
df = df[(df['circle_x'] >= 0) & (df['circle_y'] < 300)]
df['dx'] = df['enemy_x'] - df['my_x']
df['dy'] = df['enemy_y'] - df['my_y']
df['daz'] = df['enemy_az'] - df['my_az']
df.loc[(df['daz'] <= -180), 'daz'] = df['daz'] + 360
df.loc[(df['daz'] >   180), 'daz'] = df['daz'] - 360
df.to_csv(sys.stdout, float_format='%.6f')
