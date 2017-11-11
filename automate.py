"""
This script automates the different combinations of suppliers for the trainings.
"""

from script import supp_l as k

import sys
arg1 = int(sys.argv[1])  # j
arg2 = int(sys.argv[2])  # i

f = open('constants.py', 'w')

def skip(m, n):
    f.seek(0)  # to empty the file
    # f.write("SUPP_ORIGIN = " + str(k) + "\n")
    if n >= 0: # check if it's a 2-supplier-removal run
        del(k[int(n)])
    if m >= 0:
        del(k[int(m)])
    k.append('S-others')
    f.write("SUPP_PREPRO = " + str(k) + "\n")

skip(arg1, arg2)

if arg2 >= 0:
    f.write("OUT_NAME = 'out " + str(arg1) + " and " + str(arg2) + ".csv'\n")
elif arg1 >= 0:
    f.write("OUT_NAME = 'out " + str(arg1) + ".csv'\n")
else:
    f.write("OUT_NAME = 'out.csv'\n")

f.close()
