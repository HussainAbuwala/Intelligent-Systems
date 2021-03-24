#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:18:47 2019
Reference taken from the website below:
https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
@author: hussainabuwala
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#fuzzy membership function generation

type_of_dirt_input = 43
dirtiniess_input = 67


type_of_dirt = np.arange(0, 101, 1)
dirtiness  = np.arange(0, 101, 1)
wash_time = np.arange(0,61,1)


dirt_not_greasy = fuzz.trimf(type_of_dirt, [0, 10, 50])
dirt_moderate_greasy = fuzz.trimf(type_of_dirt, [10, 50, 90])
dirt_extreme_greasy = fuzz.trimf(type_of_dirt, [50, 100, 100])

dirtiness_small = fuzz.trimf(dirtiness, [0, 10, 50])
dirtiness_medium = fuzz.trimf(dirtiness, [10, 50, 90])
dirtiness_large = fuzz.trimf(dirtiness, [50, 100, 100])

wash_time_vs = fuzz.trimf(wash_time, [0, 0, 12])
wash_time_s =  fuzz.trimf(wash_time, [0, 15, 20])
wash_time_m = fuzz.trimf(wash_time, [12, 27, 42])
wash_time_l = fuzz.trimf(wash_time, [20, 38, 60])
wash_time_vl = fuzz.trimf(wash_time, [42, 60, 60])


fig, (ax1, ax2,ax3) = plt.subplots(nrows=3, figsize=(8, 13))

ax1.plot(type_of_dirt, dirt_not_greasy, 'b', linewidth=1.5, label='Not greasy')
ax1.plot(type_of_dirt, dirt_moderate_greasy, 'g', linewidth=1.5, label='Moderately Greasy')
ax1.plot(type_of_dirt, dirt_extreme_greasy, 'r', linewidth=1.5, label='Extremely Greasy')
ax1.set_title('Type of dirt')
ax1.legend()


ax2.plot(dirtiness, dirtiness_small, 'b', linewidth=1.5, label='Low')
ax2.plot(dirtiness, dirtiness_medium, 'g', linewidth=1.5, label='Medium')
ax2.plot(dirtiness, dirtiness_large, 'r', linewidth=1.5, label='High')
ax2.set_title('Degree of Dirtiness')
ax2.legend()

ax3.plot(wash_time, wash_time_vs, 'b', linewidth=1.5, label='Very Short')
ax3.plot(wash_time, wash_time_s, 'g', linewidth=1.5, label='Short')
ax3.plot(wash_time, wash_time_m, 'r', linewidth=1.5, label='Medium')
ax3.plot(wash_time, wash_time_l, 'y', linewidth=1.5, label='Long')
ax3.plot(wash_time, wash_time_vl, 'm', linewidth=1.5, label='Very Long')
ax3.set_title('Wash Time')
ax3.legend()


# Turn off top/right axes
for ax in (ax1, ax2,ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()



#-----------------------------------------------------------------------------
#Rule evaluation


'''
Dirt_type       Dirtiness      Wash Time
-------------------------------------------
Not greasy	 &  Small	   |   Very Short         Rule (1)
Not greasy	 &  Medium	   |   Short              Rule (2)
Not greasy	 &  Large	   |   Medium             Rule (3)
Medium	     &  Small	   |   Medium             Rule (4)
Medium	     &  Medium	   |   Long               Rule (5)
Medium	     &  Large	   |   Long               Rule (6)
Greasy	     &  Small	   |   Medium             Rule (7)
Greasy	     &  Medium	   |   Long               Rule (8)
Greasy	     &  Large	   |   Very Long          Rule (9)
'''


tod_not_greasy_val = fuzz.interp_membership(type_of_dirt, dirt_not_greasy, type_of_dirt_input)
tod_medium_val = fuzz.interp_membership(type_of_dirt, dirt_moderate_greasy, type_of_dirt_input)
tod_greasy_val = fuzz.interp_membership(type_of_dirt, dirt_extreme_greasy, type_of_dirt_input)


dirtiness_small_val = fuzz.interp_membership(dirtiness, dirtiness_small, dirtiniess_input)
dirtiness_medium_val = fuzz.interp_membership(dirtiness, dirtiness_medium, dirtiniess_input)
dirtiness_large_val = fuzz.interp_membership(dirtiness, dirtiness_large, dirtiniess_input)




rule1 = np.fmin(tod_not_greasy_val, dirtiness_small_val)
wash_time_very_short_val = np.fmin(rule1, wash_time_vs)

rule2 = np.fmin(tod_not_greasy_val, dirtiness_medium_val)
wash_time_short_val = np.fmin(rule2, wash_time_s)

rule3 = np.fmin(tod_not_greasy_val, dirtiness_large_val)
rule4 = np.fmin(tod_medium_val, dirtiness_small_val)
rule7 = np.fmin(tod_greasy_val, dirtiness_small_val)
wash_time_medium_val = np.fmin(np.fmax(rule3,np.fmax(rule4,rule7)),wash_time_m)

rule5 = np.fmin(tod_medium_val, dirtiness_medium_val)
rule6 = np.fmin(tod_medium_val, dirtiness_large_val)
rule8 = np.fmin(tod_greasy_val, dirtiness_medium_val)
wash_time_long_val = np.fmin(np.fmax(rule5,np.fmax(rule6,rule8)),wash_time_l)

rule9 = np.fmin(tod_greasy_val, dirtiness_large_val)
wash_time_very_long_val = np.fmin(rule9, wash_time_vl)



wash_time_zeroes = np.zeros_like(wash_time)
print(wash_time_zeroes)





# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(wash_time, wash_time_zeroes, wash_time_very_short_val, facecolor='b', alpha=0.7)
ax0.plot(wash_time, wash_time_vs, 'b', linewidth=0.5, linestyle='--', )

ax0.fill_between(wash_time, wash_time_zeroes, wash_time_short_val, facecolor='g', alpha=0.7)
ax0.plot(wash_time, wash_time_s, 'g', linewidth=0.5, linestyle='--')

ax0.fill_between(wash_time, wash_time_zeroes, wash_time_medium_val, facecolor='r', alpha=0.7)
ax0.plot(wash_time, wash_time_m, 'r', linewidth=0.5, linestyle='--')

ax0.fill_between(wash_time, wash_time_zeroes, wash_time_long_val, facecolor='y', alpha=0.7)
ax0.plot(wash_time, wash_time_l, 'y', linewidth=0.5, linestyle='--')

ax0.fill_between(wash_time, wash_time_zeroes, wash_time_very_long_val, facecolor='m', alpha=0.7)
ax0.plot(wash_time, wash_time_vl, 'm', linewidth=0.5, linestyle='--')

ax0.set_title('Output membership activity')


# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()








# Aggregate all three output membership functions together
ag1 = np.fmax(wash_time_medium_val,
                     np.fmax(wash_time_long_val, wash_time_very_long_val))

final_aggregation = np.fmax(wash_time_very_short_val,np.fmax(wash_time_short_val,ag1))

# Calculate defuzzified result
wt = fuzz.defuzz(wash_time, final_aggregation, 'centroid')
wt_activation = fuzz.interp_membership(wash_time, final_aggregation, wt)  # for plot

print(wt)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(wash_time, wash_time_vs, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(wash_time, wash_time_s, 'g', linewidth=0.5, linestyle='--')
ax0.plot(wash_time, wash_time_m, 'r', linewidth=0.5, linestyle='--')
ax0.plot(wash_time, wash_time_l, 'y', linewidth=0.5, linestyle='--')
ax0.plot(wash_time, wash_time_vl, 'm', linewidth=0.5, linestyle='--')

ax0.fill_between(wash_time, wash_time_zeroes, final_aggregation, facecolor='Orange', alpha=0.7)
ax0.plot([wt, wt], [0, wt_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
