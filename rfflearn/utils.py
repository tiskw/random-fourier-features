#!/usr/bin/env python3
#
# Collection of utility functions used in the example code of this module.
#
# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# Date  : January 29, 2021
######################################### SOURCE START ########################################

import time

### Class for measure elasped time using 'with' sentence.
class Timer:

    def __init__(self, message = "", unit = "s", devide_by = 1):
        self.message   = message
        self.time_unit = unit
        self.devide_by = devide_by

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, ex_type, ex_value, trace):
        dt = (time.time() - self.t0) / self.devide_by
        if   self.time_unit == "ms": dt *= 1E3
        elif self.time_unit == "us": dt *= 1E6
        print("%s%f [%s]" % (self.message, dt, self.time_unit))

### Map a real value in [0, 1] to a string color code according to the JET color map.
### For example, colormap_jet(0.25) -> "#008080".
def colormap_jet(value):

    if value < 0.0:
        r = round(0)
        g = round(0)
        b = round(255)
    elif value < 0.5:
        value = 2 * value
        r = round(0)
        g = round(255 * value)
        b = round(255 * (1 - value))
    elif value < 1.0:
        value = 2 * value - 1
        r = round(255 * value)
        g = round(255 * (1 - value))
        b = round(0)
    else:
        r = round(255)
        g = round(0)
        b = round(0)

    return ("#%02x%02x%02x" % (r, g, b))

######################################### SOURCE FINISH #######################################
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
