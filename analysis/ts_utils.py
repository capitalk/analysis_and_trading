import numpy as np
import logging as log


MILLIS_DAY = 24*60*60*1000;

def millis_to_hmsms(millis):
    seconds = millis/1000
    h = seconds / 3600 
    m = (seconds - (h*3600))/60
    ss = (seconds - ((h*3600)+(m*60)))
    ms = millis - ((ss*1000)+(m*60*1000)+(h*3600*1000))
    if h > 24 or m > 60 or ss > 60 or ms > 1000: 
        raise RuntimeError("Invalid time: " , h , ':' , m , ':' , ss , '.' , ms)

    return (h, m, ss, ms)

