
from __future__ import with_statement
import threading
import sys


# Implementation of Ticker class
class Ticker(threading.Thread):
    def __init__(self, msg, breakAfter=True):
	threading.Thread.__init__(self)
	self.msg = msg
	self.event = threading.Event()
        self.break_after = breakAfter 
    def __enter__(self):
	self.start()
    def __exit__(self, ex_type, ex_value, ex_traceback):
	self.event.set()
	self.join()
    def run(self):
	sys.stdout.write(self.msg)
	while not self.event.isSet():
	    sys.stdout.write(".")
	    sys.stdout.flush()
	    self.event.wait(1)
        if self.break_after:
            sys.stdout.write("\n")


#pipeline_nesting = 0

def mk_pipeline(*functions):
    #tabs = "\t" * pipeline_nesting
    #global pipleline_nesting
    #pipeline_nesting += 1
    def compute(data):
        for fn in functions:
            msg = "[pipeline] Executing " + fn.__module__ +  " : " +  fn.__name__
            with Ticker(msg):
                data = fn(data)
        return data
#    pipeline_nesting -= 1
    return compute
    
