
import numpy as np 
import os 
import h5py 

def search(hdf, print_limit = None):
  bids = hdf['bid/1ms'][:]
  bid_vol = hdf['bid_vol/1ms'][:]
  t = hdf['t/1ms'][:]
  offers = hdf['offer/1ms'][:]
  offer_vol = hdf['offer_vol/1ms'][:]
  mask = bids > offers
  idx = np.nonzero(mask)[0]
  assert( (len(t)-1) not in idx)
  durs = t[idx+1] - t[idx]  
  delta = bids[idx] - offers[idx]
  prct = delta / bids[idx]
  size = np.minimum(bid_vol[idx], offer_vol[idx])
  scores = (durs > 10) * prct * size
  print "Total = %d, count(dur > 10, prct >= 1 pip) = %d" % (len(scores), np.sum( (durs > 10) & (prct >= 10 ** -4)))
  print "%12s %12s %12s %12s %12s %12s" % \
    ("dur", "bid", "offer", "cross", "prct of bid", "min. vol.")
  for counter, i in enumerate(reversed(np.argsort(scores))):
    if print_limit is not None and counter >= print_limit:
      break
    print "%12d %12s %12s %12.6f %12.6f %12d" % \
      (durs[i], bids[i], offers[i], delta[i], prct[i], size[i])
  return idx, delta, prct, size, durs, scores 

def search_dir(path, print_limit = None):
  all_prcts = []
  for fname in os.listdir(path):
    if fname.endswith(".hdf"):
      print fname
      f = h5py.File(os.path.join(path, fname))
      _, _, prct, _, durs, _ = search(f, print_limit) 
      if print_limit is not None and print_limit > 0:
        print
      for p,d in zip(prct, durs):
        if d > 10: all_prcts.append(p)
  return np.array(all_prcts)    

