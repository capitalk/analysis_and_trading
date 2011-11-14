import cloud 
import boto 
import os 
import tempfile 
from dataset import Dataset 

AWS_ACCESS_KEY_ID = 'AKIAITZSJIMPWRM54I4Q' 
AWS_SECRET_ACCESS_KEY = '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' 


def get_hdf_bucket(bucket='capk-fxcm'):
    # just in case
    import socket
    socket.setdefaulttimeout(None)
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    return conn.get_bucket(bucket)


def load_s3_file(filename, max_failures=2, cache_dir='/s3_cache'):     
    print "Loading", filename, "from S3"
    cache_name = cache_dir + "/" + filename 
    if os.path.exists(cache_name): 
        print "Found", filename, "in environment's local cache, returning", cache_name 
        return cache_name 
    max_failures = 3 
    fail_count = 0 
    got_file = False 
    while fail_count < max_failures and not got_file:
        bucket = get_hdf_bucket()
        key = bucket.get_key(filename) 
        if key is None: raise RuntimeError("file not found: " + filename)
        (fileid, local_filename) = tempfile.mkstemp(prefix=filename)
        print "Created local file:", local_filename 
        print "Copying data from S3"
        try: 
            key.get_contents_to_filename(local_filename)
            got_file = True 
        except: 
            print "Download from S3 failed" 
            fail_count += 1
            os.remove(local_filename) 
            if fail_count >= max_failures: raise 
            else: print "...trying again..."
    return local_filename 
    
def load_s3_files(filenames):
    return [Dataset(load_s3_file(remote_filename)) for remote_filename in filenames]

def load_s3_files_in_parallel(filenames):
    jobids = cloud.mp.map(load_s3_file, filenames)
    local_filenames = cloud.mp.result(jobids)
    return [Dataset(f) for f in local_filenames]

def print_s3_hdf_files(): 
    bucket = get_hdf_bucket()
    filenames = [k.name for k in bucket.get_all_keys() if k.name.endswith('hdf')]
    print "\n".join(filenames )

def make_s3_filenames(ecn, ccy, dates): 
    assert ecn is not None
    assert ccy is not None 
    assert dates is not None and len(dates) > 0 
    
    ecn = ecn.upper()
    ccy = ccy.upper().replace('/', '') 
    dates = [d.replace('/', '_') for d in dates]
    return [ecn +"_" + ccy + "_" + d + ".hdf" for d in dates] 
