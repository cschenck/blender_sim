#!/usr/bin/env python

import signal
import subprocess
import time

def dont_ingore_signals():
    signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    signal.signal(signal.SIGCONT, signal.SIG_DFL)
proc = subprocess.Popen(['top'], preexec_fn=dont_ingore_signals)

while True:
	print("master")
	time.sleep(1.0)