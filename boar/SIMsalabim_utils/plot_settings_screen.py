######################################################################
#################### Default Settings matplotlib #####################
######################################################################
# Author: Vincent M. Le Corre
# Github: https://github.com/VMLC-PV

# Import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt 
# import 


mpl.rcParams['savefig.dpi'] = 300
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

line_thickness = 2# default = 1.1
# print(plt.rcParams.keys())

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', linewidth=line_thickness, titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', top=True, bottom=True, direction='in', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('xtick.major', size=line_thickness * 6.25, width=line_thickness,pad =10)
plt.rc('xtick.minor', size=line_thickness * 2.5, width=line_thickness ,pad =10)
plt.rc('ytick', left=True, right=True, direction='in', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick.major', size=line_thickness * 6.25, width=line_thickness)
plt.rc('ytick.minor', size=line_thickness * 2.5, width=line_thickness)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=5)  # fontsize of the figure title
plt.rc('lines', linewidth=3)
#set figure size
# plt.rcParams['figure.figsize'] = [12, 10]
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
