# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:31:24 2020

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
1. Iteration is much faster than recursion
...............................................................
"""


#...................................
#    Algorithms
#...................................
#    - Fibonacci
#...................................
import timeit
import numpy as np
import pandas as pd
import numba as nb
import datetime as dt
from time import perf_counter_ns
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

# ........ METHOD = Simple, recursive ..................
def fib_rec_py1( n ):
        if n < 2:
                return n
        else:
                return fib_rec_py1( n - 1) + fib_rec_py1( n - 2)                
#timeit.repeat( "for x in range(20): fib_rec_py1(x)", "from __main__ import fib_rec_py1", number = 10 )
                
        
# ........ METHOD = Numba - dynamic compiling ..................
fib_rec_nb = nb.jit(fib_rec_py1)

# ........ METHOD = Caching of intermediary results.............
from functools import lru_cache as cache
@cache( maxsize = None )
def fib_rec_py2( n ):
        if n < 2:
                return n
        else:
                return fib_rec_py2( n - 1) + fib_rec_py2( n - 2)                

               
arr = np.empty( ( 0, 2 ), int )        
arr_nb = np.empty( ( 0, 2 ), int )        
arr_ch = np.empty( ( 0, 2 ), int )        

res = fib_rec_nb( 2 )

min_b = 10
max_b = 40
for k in range( min_b, max_b + 1, 1):
        # ..........................................        
        t0 = perf_counter_ns()
        res = fib_rec_py1( k )
        t1 = perf_counter_ns() - t0 
        arr = np.append( arr, np.array([[ k, t1 ]]), axis = 0 )
        # ..........................................
        t0 = perf_counter_ns()
        res = fib_rec_nb( k )
        t1 = perf_counter_ns() - t0 
        arr_nb = np.append( arr_nb, np.array([[ k, t1 ]]), axis = 0 )
        # ..........................................
        t0 = perf_counter_ns()
        res = fib_rec_py2( k )
        t1 = perf_counter_ns() - t0 
        arr_ch = np.append( arr_ch, np.array([[ k, t1 ]]), axis = 0 )
        # ..........................................
        print( "At k = " + str( k ) + ", time = " + str( dt.datetime.now() ) )
        

df_rec = pd.DataFrame( arr, columns = ['k', 'Time(fibo)'] )
df_rec['Type'] = len( df_rec.index ) * ['Recursive']

df_rec_nb = pd.DataFrame( arr_nb, columns = ['k', 'Time(fibo)'] )
df_rec_nb['Type'] = len( df_rec_nb.index ) * ['Numba - dynamic compiling']

df_rec_ch = pd.DataFrame( arr_ch, columns = ['k', 'Time(fibo)'] )
df_rec_ch['Type'] = len( df_rec_ch.index ) * ['Caching of intermediary results']

df = df_rec.append( df_rec_nb )
df = df.append( df_rec_ch )
df['sec'] = df['Time(fibo)'] / 10**9


grouped = df.groupby(['Type'])
fig, ax = plt.subplots()
for key, group in grouped:
        group.plot( 'k','sec', marker='o', linestyle='--',label = key, ax = ax)
        
plt.legend( loc = 'best' )        
plt.title('Fibbonaci')
plt.show()


# ........ METHOD = Simple, iterative ..................
def fib_it_py( n ):
        x,y = 0,1
        for i in range( 1,n + 1):
                x, y = y, x + y
        return x

# ........ METHOD = Numba( dynamic compiling ) of iterative ........
fib_it_nb = nb.jit( fib_it_py )

arr = np.empty( ( 0, 2 ), int )        
arr_nb = np.empty( ( 0, 2 ), int )        
arr_ch = np.empty( ( 0, 2 ), int )        

res = fib_it_nb( 2 )

min_b = 10
max_b = 200
for k in range( min_b, max_b + 1, 1):
        # ..........................................        
        t0 = perf_counter_ns()
        res = fib_it_py( k )
        t1 = perf_counter_ns() - t0 
        arr = np.append( arr, np.array([[ k, t1 ]]), axis = 0 )
        # ..........................................
        t0 = perf_counter_ns()
        res = fib_it_nb( k )
        t1 = perf_counter_ns() - t0 
        arr_nb = np.append( arr_nb, np.array([[ k, t1 ]]), axis = 0 )
        # ..........................................        
        t0 = perf_counter_ns()
        res = fib_rec_py2( k )
        t1 = perf_counter_ns() - t0 
        arr_ch = np.append( arr_ch, np.array([[ k, t1 ]]), axis = 0 )
        # ..........................................        
        if( k % 5 == 0):
                print( "At k = " + str( k ) + ", time = " + str( dt.datetime.now() ) )
       

df_it_simple = pd.DataFrame( arr, columns = ['k', 'Time(fibo)'] )
df_it_simple['Type'] = len( df_it_simple.index ) * ['Iterative']

df_it_nb = pd.DataFrame( arr_nb, columns = ['k', 'Time(fibo)'] )
df_it_nb['Type'] = len( df_it_nb.index ) * ['Numba - dynamic compiling']

df_rec_ch2 = pd.DataFrame( arr_ch, columns = ['k', 'Time(fibo)'] )
df_rec_ch2['Type'] = len( df_rec_ch2.index ) * ['RECURSIVE: Caching of intermediary results']


df_it = df_it_simple.append( df_it_nb )
df_it = df_it.append( df_rec_ch2 )
df_it['sec'] = df_it['Time(fibo)'] / 10**9


grouped = df_it.groupby(['Type'])
fig, ax = plt.subplots()
for key, group in grouped:
        group.plot( 'k','sec', marker='o', linestyle='--',label = key, ax = ax)
        
plt.legend( loc = 'best' )        
plt.title('Fibbonaci')
plt.show()          