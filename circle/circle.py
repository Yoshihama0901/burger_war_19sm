#!/usr/bin/env python
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], header=None)
df.columns = ['my_x', 'my_y', 'enemy_x', 'enemy_y', 'circle_x', 'circle_y', 'circle_r']
df = df[(df['circle_x'] >= 0) & (df['circle_y'] < 300)]
df.to_csv(sys.stdout, float_format='%.6f')
