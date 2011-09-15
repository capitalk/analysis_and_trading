#!/usr/bin/env python

"""Test script fo remote process start
"""

import time
import sys

def main():
	f = open("./FINISHED", "w")
	time.sleep(30)
	f.write("\n")
	f.close()

	
if  __name__ == "__main__":
	main()
