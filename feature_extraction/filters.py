
import numpy 

# assume timestamp is in field named 't' 
def drop_out_of_order(features):
    good_mask = features.t[1:] > features.t[:-1]  
    good_indices = numpy.nonzero(good_mask)[0] + 1
    features.transform_values(lambda xs: xs[good_indices])
    return features

def diff(timeseries):
  return timeseries[1:] - timeseries[:-1] 

def calc_derivs(features):
    for name,timeseries in features.items():
        deriv = diff(timeseries)
        features[name] = timeseries[1:]
        features[name + "_deriv"] = deriv
        
        #accel = diff(deriv)
        ## shorten original data by 2 samples
        #setattr(features, name, timeseries[2:])
        ## shorten derivative by 1 sample 
        #setattr(features, name + "_deriv", deriv[1:])
        #setattr(features, name + "_accel", accel)
    return features

def pairwise_prod(tsc):
    for name1, timeseries1 in tsc.items():
        for name2, timeseries2 in tsc.items(): 
            key = name1 + "_times_" + name2
            tsc[key] = timeseries1 * timeseries2
    return tsc
