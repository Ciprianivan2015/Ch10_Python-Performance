# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:58:33 2020

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
Brownian motion: THEORY (short)
...............................................................
Stochastic equation:
.....................
dSt = r*St*dt + sigma * St * dZt

Discretized over equidistant time intervals dt
............................................
St = S(t-dt) * exp( (r-sigma^2/2) * dt + sigma * sqrt( dt ) * z )

Call (European)
.................
C0 = exp( -rT ) 1 / N sum{over N} { max( S_{T} - K,0 ) }
"""
import numpy as np
import math 
from pylab import plt, mpl

plt.style.use('seaborn')

# .... nr paths .......
N = 10 **4
# .... nr interval ...
M = 250

S0 = 100.0
T = 1.0
r = 0.02
sigma = 0.20
K = 110


def mcs_simulation_py( p ):
        M, N, S0 = p
        dt = T / M
        S = np.zeros( ( M + 1, N) )
        S[ 0 ] = S0
        rn = np.random.standard_normal ( S.shape )
        # ... for each sub-interval of Time .... 
        for t in range( 1, M + 1):
                # for each path 
                for k in range( N ):
                        expr = ( r - sigma ** 2 / 2 ) * dt + sigma * math.sqrt( dt ) * rn[ t, k ]
                        S[ t, k] = S[ t - 1, k ] * math.exp( expr )
        return S

%time S = mcs_simulation_py( ( M, N, S0 ) )
S[ -1 ].mean()
S0 * math.exp( r * T )

# .... Call price .....
C0 = math.exp( -r * T ) * np.maximum( S[ -1 ] - K, 0 ).mean()
# .... Put price .....
P0 = math.exp( -r * T ) * np.maximum( K - S[ -1 ], 0 ).mean()

# ................................................................
# 
# ................................................................
plt.hist( S[-1], bins = 100, edgecolor = 'darkblue', color = 'darkgray' )
plt.vlines( x = S[ -1 ].mean(), ymin = 0, ymax = 450, 
           linestyle ='dotted', color = 'darkred' )
plt.vlines( x = S[ 0 ], ymin = 0, ymax = 450, 
           linestyle ='dotted', color = 'darkgreen' )
