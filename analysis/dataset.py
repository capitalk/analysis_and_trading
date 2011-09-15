import h5py
import numpy as np 
    
        
class Dataset():
    def __init__(self, filename):
        self.filename = filename 
        hdf = h5py.File(filename, 'r')
        self.hdf = hdf 
        self.t = hdf['t/100ms'][...]
        self.indices = np.arange(len(self.t))
        self.features = list(hdf.attrs['features'])
        self.reducers = list(hdf.attrs['reducers'])
        self.timescales = hdf.attrs['timescales']
        
        # if the HDF file doesn't tell us which currency it is, 
        # try to infer it from the file name 
        if "currency_pair" in hdf.attrs:
            self.currency_pair = hdf.attrs["currency_pair"]
        else:
            currency_names = [
                "AUD", "CAD", "EUR", "GBP", "NZD", "JPY", "EUR", "CHF" 
            ] 
            self.currency_pair = None 
            for part in filename.split("_"):
                for currency in currency_names:
                    if currency in part:
                        self.currency_pair = part[0:3], part[3:6] 
            assert self.currency_pair is not None 
        #self.no_timescale = ['t', 'idx', 't_deriv']
        # features which don't need aggregators at the end
        #self.no_aggregator = [
        #    't',  'idx', 't_deriv', 'count', 'count_deriv'
        #]

    
    # either return the dataset or return None
    def find_path(self, path, start_idx=None, end_idx=None): 
        if path == 'idx':
            return self.indices[start_idx:end_idx]
        else:
            if path in self.hdf:
                d = self.hdf[path]
                # if given a group, try traversing til you find a dataset
                if type(d) == h5py.Group:
                    for elt in d.values():
                        if type(elt) == h5py.Dataset:
                            print "[find_path] Given",  path, "inferring", elt.name
                            d = elt
                            break
                    
                return d[start_idx:end_idx]
            else: 
                # try splitting path and inserting a reducer 
                # e.g. bid/slope -> bid/mean/slope
                # ...and then try this function again to ultimately find
                # bid/mean/slope/100ms
                parts = [part for part in path.split('/') if len(part) > 0]
                
                has_reducer = False
                for r in self.reducers:
                    if r in parts: has_reducer = True 
                if not has_reducer:
                    default_reducer = 'mean'
                    path2 = '/'.join([parts[0], default_reducer] + parts[1:])
                    print "[find_path] Couldn't find", path, "trying", path2 
                    return self.find_path(path2, start_idx, end_idx)
                else: 
                    return None

    # either return the dataset or throw an error 
    def get_col(self, path, start_idx=None, end_idx=None):
        d = self.find_path(path, start_idx, end_idx)
        if d is not None: return d
        else: raise RuntimeError("[Dataset] Not found: " + path )
            
    
    
        
            
    def __getitem__(self, s):
        return self.get_col(s)
            
             
    # list containing either feature names or pairs of feature names + 
    # normalization functions 
    def get_cols(self, cols):
        nrows = len(self.t)
        ncols = len(cols)
        data = np.zeros((nrows, ncols))
        colidx = 0 
        colnames = [] 
        for elt in cols:
            if type(elt) == 'tuple':
                name, normalizer = elt
                colnames.append(name)
                vec = normalizer(self.get_col(name))
            else:
                colnames.append(elt)
                vec = self.get_col(elt)
                
            data[:, colidx] = vec
            colidx += 1
        return data, colnames
        
    def get_timescales(self, ts, names=None, features=None, reducers=None, start_idx=None, end_idx=None):
        if names is None: names = [] 
        if reducers or features:
            if features is None: features = self.features
            if reducers is None: reducers = self.reducers
            
            for f in features:
                for r in reducers:
                    names.append(f + "/" + r)
        
        
        cols = [] 
        for name in names:
            for t in ts:
                cols.append(name + "/scale" + str(t))
        
        ncols= len(cols)
        if start_idx and end_idx:
            nrows = end_idx - start_idx
        else:
            nrows = len(self.indices)
        result = np.zeros((nrows, ncols))
                
        colidx = 0
        for col in cols:
            
            result[:, colidx] = self.get_col(col, start_idx, end_idx)
            colidx += 1
        return result, cols 
        
            
