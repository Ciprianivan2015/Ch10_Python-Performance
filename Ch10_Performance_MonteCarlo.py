# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 18:02:14 2020

@author: Cipriandan
...............................................................
- Performance in Python
...............................................................
        1. Vectorization
        2. Dynamic compiling
        3. Static compiling
        4. Multiprocessing
        
        Topics
        ...................
        1. Loops
        2. Algorithms
        3. Binomial trees
        4. Monte Carlo Simulation
        5. Recursive pandas: EWMA        
...............................................................
"""

import random as rd
import numpy as np
import pandas as pd
import numba as nb
import datetime as dt
from time import perf_counter_ns

from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

# ..............................................................
# ............ Subject exploration .............................
# ..............................................................
rn = [ ( rd.random() * 2 - 1, rd.random() * 2 - 1 ) for _ in range( 5000 )]
rn = np.array( rn )

distance = np.sqrt( ( rn**2 ).sum( axis = 1 ) )
frac = ( distance <= 1.0 ).sum( ) / len( distance )
pi_mcs = frac * 4

#....... PLOT .............................
fig = plt.figure( figsize = ( 8, 8 ) )
ax = fig.add_subplot( 1, 1, 1 )
circ = plt.Circle( (0,0), radius = 1, edgecolor = 'g', lw = 2.0, facecolor = 'None' )
box = plt.Rectangle( (-1,-1), 2,2, edgecolor = 'b', alpha = 0.3 )
ax.add_patch( circ )
ax.add_patch( box )

plt.plot( rn[ :, 0], rn[:, 1 ], 'r.'  )
plt.xlim( -1.1, 1.1)
plt.ylim( -1.1, 1.1)

# ..............................................................
# ............ MonteCarlo function: NumPy.......................
# ..............................................................
def mcs_pi_np( n ):
        rn = [ ( rd.random() * 2 - 1, rd.random() * 2 - 1 ) for _ in range( n )]
        rn = np.array( rn )        
        distance = np.sqrt( ( rn**2 ).sum( axis = 1 ) )
        frac = ( distance <= 1.0 ).sum( ) / n
        return frac * 4        

# ................................................................
# ............ MonteCarlo function: for loop, no vectorization ...
# ................................................................
def mcs_pi_py( n ):
        circle_dots = 0
        for _ in range( n ):
                x, y = rd.random(),rd.random()
                if ( x ** 2 + y ** 2)**0.5 <= 1:
                        circle_dots += 1
        frac = circle_dots / n                       
        return frac * 4

# ................................................................
# ............ MonteCarlo function: Dynamic compiling ............
# ................................................................
mcs_pi_nb1 = nb.jit( mcs_pi_np )    # ... issues of deprecation  
mcs_pi_nb2 = nb.jit( mcs_pi_py )      

res2 = mcs_pi_nb1( 50 )  
res2 = mcs_pi_nb2( 50 )  
%timeit res = mcs_pi_np( 500000 )        
%timeit res1= mcs_pi_py( 500000 )  
%timeit res2= mcs_pi_nb1( 500000 )  
%timeit res2= mcs_pi_nb2( 500000 )  


# ............ Study ................
arr = np.empty( (0,3), float )
myr = np.arange( 1, 9 + 1, 1)
ml  = [10**6] * myr
mlt = [10**7] * myr
mlh = [10**8] * myr
ss  = np.append( ml, mlt )
ss  = np.append( ss, mlh ) 

sci_pi = np.pi

for k in ss:        
        t0 = perf_counter_ns()
        my_pi = mcs_pi_nb2( k )
        t1 = perf_counter_ns() - t0
        arr = np.append( arr, np.array([[ k, my_pi, t1 ]]), axis = 0 )
        print( "At k = " + str( k ) + ", time = " + str( dt.datetime.now() ) )        
        
df = pd.DataFrame( arr, columns = ['SampleSize', 'Estim{ Pi }', 'Estim_Time'] )

plt.scatter( df['SampleSize'], df['Estim{ Pi }'], s = np.log( df['Estim_Time']) )
plt.hlines( y = sci_pi, xmin = 0, xmax = max( ss ), linestyles = 'dashed' , color = 'r', alpha = 0.3 )
plt.title('Monte Carlo Simulation: estimation of PI')
plt.xlabel( 'SampleSize')