#!/usr/bin/env python

import os
import site

packdir = site.getsitepackages()[0]

try:
    f = open(os.path.join(packdir, "import_cutil.pth"), "w")
    f.write(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    f.flush()
    f.close()
except IOError:
    print("ERROR: Please make sure you are root or have write permissions in %s." % packdir)
    print("Run \'sudo ./install.py\' to install correctly.")
