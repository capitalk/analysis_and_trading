import features 
from featurePipeline import FeaturePipeline
from optparse import OptionParser
import os, os.path
import h5py, gzip, datetime
import boto
import fnmatch 
import progressbar  
import cloud

cloud.setkey(2579, 'f228c0325cf687779264a0b0698b0cfe40148d65')

extractor = FeaturePipeline()
extractor.add_feature('t', features.millisecond_timestamp)
extractor.add_feature('bid', features.best_bid)
extractor.add_feature('offer', features.best_offer)
extractor.add_feature('bid_range', features.bid_range)
extractor.add_feature('offer_range', features.offer_range)

extractor.add_feature('spread', features.spread)

extractor.add_feature('locked', features.locked)
extractor.add_feature('crossed', features.crossed)

extractor.add_feature('midprice', features.midprice)
extractor.add_feature('bid_vwap', features.bid_vwap)
extractor.add_feature('offer_vwap', features.offer_vwap)

extractor.add_feature('bid_slope', features.bid_slope)
extractor.add_feature('offer_slope', features.offer_slope)

extractor.add_feature('offer_vol', features.best_offer_volume)
extractor.add_feature('bid_vol', features.best_bid_volume)

extractor.add_feature('total_bid_vol', features.bid_volume)
extractor.add_feature('total_offer_vol', features.offer_volume)
extractor.add_feature('t_mod_1000', features.fraction_of_second, use_window_reducers=False)
extractor.add_feature('message_count', features.message_count, sum_100ms=True)
# V3 orderbook action  features
extractor.add_feature('bid_tr8dr', features.bid_tr8dr)
extractor.add_feature('offer_tr8dr', features.offer_tr8dr)
extractor.add_feature('tr8dr', features.tr8dr)

extractor.add_feature('added_total_bid_vol', features.added_bid_volume, sum_100ms=True)
extractor.add_feature('added_total_bid_count', features.added_bid_count, sum_100ms=True)
extractor.add_feature('added_total_offer_vol', features.added_offer_volume, sum_100ms=True)
extractor.add_feature('added_total_offer_count', features.added_offer_count, sum_100ms=True)

extractor.add_feature('added_best_bid_vol', features.added_best_bid_volume, sum_100ms=True)
extractor.add_feature('added_best_bid_count', features.added_best_bid_count, sum_100ms=True)
extractor.add_feature('added_best_offer_vol', features.added_best_offer_volume, sum_100ms=True)
extractor.add_feature('added_best_offer_count', features.added_best_offer_count, sum_100ms=True)

extractor.add_feature('deleted_total_bid_vol', features.deleted_bid_volume, sum_100ms=True)
extractor.add_feature('deleted_total_bid_count', features.deleted_bid_count, sum_100ms=True)
extractor.add_feature('deleted_total_offer_vol', features.deleted_offer_volume, sum_100ms=True)
extractor.add_feature('deleted_total_offer_count', features.deleted_offer_count, sum_100ms=True)

extractor.add_feature('filled_bid_vol', features.filled_bid_volume, sum_100ms=True)
extractor.add_feature('filled_bid_count', features.filled_bid_count, sum_100ms=True)
extractor.add_feature('filled_offer_vol', features.filled_offer_volume, sum_100ms=True)
extractor.add_feature('filled_offer_count', features.filled_offer_count, sum_100ms=True)

extractor.add_feature('canceled_bid_vol', features.canceled_bid_volume, sum_100ms=True)
extractor.add_feature('canceled_bid_count', features.canceled_bid_count, sum_100ms=True)
extractor.add_feature('canceled_offer_vol', features.canceled_offer_volume, sum_100ms=True)
extractor.add_feature('canceled_offer_count', features.canceled_offer_count, sum_100ms=True)

 
 # file exists and 'finished' flag is true
def file_already_done(filename):
    print filename 
    if not os.path.exists(filename): 
      print "Doesn't exist"
      return False
    try:
        f = h5py.File(filename, 'r')
        attrs = f.attrs
        finished = 'finished' in attrs and attrs['finished']
        has_ccy = 'ccy1' in attrs and 'ccy2' in attrs 
        has_date = 'year' in attrs and 'month' in attrs and 'day' in f.attrs 
        has_venue = 'venue' in attrs 
        has_features = 'features' in attrs
        same_features = set(attrs['features']) == set(extractor.feature_fns.keys())
        if not same_features:
          print set(attrs['features']).symmetric_difference(extractor.feature_fns.keys())
        print "finished", finished
        print "date", has_date
        print "venue", has_venue
        print "has features", has_features
        print "same_features", same_features
        if not same_features:
          print f.attrs
        f.close()
        return finished
    except:
        import sys
        print sys.exc_info()
        return False

def process_local_file(input_filename, dest_1ms, dest_100ms, max_books = None, heap_profile=False):
    print "Start time:", datetime.datetime.now()
    # delete both files just in case one exists
    try:
      os.remove(dest_1ms)
      os.remove(dest_100ms)
    except OSError:
      pass
  
    # Deleted since it screws up comparisons of feature sets above 
    # and it's sort of hackish and not clearly useful 
    #if "JPY" in input_filename:
    #    extractor.add_feature(
    #      'last_bid_digit_near_zero', features.second_bid_digit_close_to_wrap)
    #    extractor.add_feature(
    #      'last_offer_digit_near_zero', features.second_offer_digit_close_to_wrap)
    #else:
    #    extractor.add_feature(
    #      'last_bid_digit_near_zero', features.fourth_bid_digit_close_to_wrap)
    #    extractor.add_feature(
    #      'last_offer_digit_near_zero', features.fourth_offer_digit_close_to_wrap)
    
    header = extractor.run(input_filename, dest_1ms, dest_100ms, max_books = max_books)

    if heap_profile:
        print "Heap contents:"
        from guppy import hpy
        heap = hpy().heap()
        print heap
        print heap[0].rp
        print heap[0].byid
        
    return header 
    
    
def output_filenames(input_path, feature_dir = None):
  assert not os.path.isdir(input_path)
  base_dir, input_name  = os.path.split(input_path)
  
  if feature_dir is None: 
    feature_dir = os.path.join(base_dir, "features")
  
  if not os.path.exists(feature_dir): 
    os.makedirs(feature_dir)
  
  no_ext = os.path.basename(input_name).split('.')[0]
  dest_1ms = os.path.join(feature_dir, no_ext + "_1ms.hdf")
  dest_100ms = os.path.join(feature_dir, no_ext + "_100ms.hdf")
  return dest_1ms, dest_100ms

def process_local_dir(input_path, output_dir = None, max_books = None, heap_profile=False):
    if not os.path.exists(input_path):
        print "Specified path does not exist: ", input_path
        exit()
        
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
        basedir = input_path
    else:
        basedir = os.path.split(input_path)[0]
        files = [os.path.basename(input_path)]
  
    count = 0
    for filename in files:
        if filename.endswith('.csv') or filename.endswith('.csv.gz'):
            count += 1
            input_filename = os.path.join(basedir, filename)
            print "----"
            print "Processing  #", count, " : ", input_filename
            
            dest_filename_1ms, dest_filename_100ms = \
              output_filenames (input_filename, output_dir)
            
            if file_already_done(dest_filename_1ms) and \
               file_already_done(dest_filename_100ms):
                print "Skipping %s found data files %s" \
                  % (input_filename, [dest_filename_1ms, dest_filename_100ms])
            else:
                process_local_file(input_filename,
                   dest_filename_1ms, dest_filename_100ms, 
                   max_books, heap_profile)
        else:
          print "Unknown suffix for", filename  


def get_s3_cxn():
  import boto 
  s3_cxn = boto.connect_s3('AKIAJSCF3K3HKREPYE6Q', 'Uz7zUOvBZzuMPLNKA2QmLaJ7lwDgJA2CYx5YZ5A0')
  if s3_cxn is None:
    raise RuntimeError("Couldn't connect to S3")
  else:
    return s3_cxn

def process_s3_file(input_bucket_name, input_key_name, output_bucket_name = None):
  if output_bucket_name is None:
     output_bucket_name = input_bucket_name + "-hdf"
  if os.access('/scratch/sgeadmin', os.F_OK | os.R_OK | os.W_OK):
    print 'Using /scratch/sgeadmin for local storage'
    tempdir = '/scratch/sgeadmin'
  elif os.access('/tmp', os.F_OK | os.R_OK | os.W_OK):
    print 'Using /tmp for local storage'
    tempdir = '/tmp'
  else:
    print 'Using ./ for local storage'
    tempdir = './'
  input_filename = os.path.join(tempdir, input_key_name)
  
  s3_cxn = get_s3_cxn() 
  in_bucket = s3_cxn.get_bucket(input_bucket_name)
  assert in_bucket is not None
  in_key = in_bucket.get_key(input_key_name)
  if in_key is None:
    raise RuntimeError(\
      "Key Not Found: bucket = " + input_bucket_name  \
      + ", input_key = " + input_key_name)
  print "Downloading", input_key_name, "to", input_filename, "..."
  if os.path.exists(input_filename) and \
     os.path.getsize(input_filename) == in_key.size:
    print "Already downloaded", input_filename, "from S3"
  else:
    in_key.get_contents_to_filename(input_filename)
  dest_1ms, dest_100ms = output_filenames(input_filename, tempdir)
  if file_already_done(dest_1ms) and file_already_done(dest_100ms):
    print "Already generated HDFs"
    f = h5y.File(dest_1ms)
    import buildBook
    a = f.attrs
    header = {
      'ccy': (a['ccy1'], a['ccy2']), 
      'year' : a['year'],
      'month' : a['month'], 
      'day' : a['day'], 
      'venue' : a['venue'], 
      #'start_time' : a['start_time'],
      #'end_time' : a['end_time'],
      'features': a['features']
    }
  else:
    print "Running feature generator..."
    header = process_local_file(input_filename, dest_1ms, dest_100ms)

  out_bucket = s3_cxn.create_bucket(output_bucket_name)
  
  print "Uploading 1ms feature file..."
  out_key_1ms = boto.s3.key.Key(out_bucket)
  out_key_1ms.key = os.path.split(dest_1ms)[1]
  out_key_1ms.set_contents_from_filename(dest_1ms)
  for k,v in header.items():
    out_key_1ms.set_metadata(k, v)
  
  print "Uploading 100ms feature file..."
  out_key_100ms = boto.s3.key.Key(out_bucket)
  out_key_100ms.key = os.path.split(dest_100ms)[1]
  out_key_100ms.set_contents_from_filename(dest_100ms)
  for k,v in header.items():
    out_key_100ms.set_metadata(k, v)
  


def process_s3_files(input_bucket_name, key_glob = '*', 
      output_bucket_name = None, use_cloud = True):
  if output_bucket_name is None:
    output_bucket_name = input_bucket_name + "-hdf"
  s3_cxn = get_s3_cxn()    
  in_bucket = s3_cxn.get_bucket(input_bucket_name)

  matching_keys = []
  for k in in_bucket:
    if fnmatch.fnmatch(k.name, key_glob):
      matching_keys.append(k.name)
     
  if use_cloud:
    
    print "Launching %d jobs" % len(matching_keys)
    
    jids = cloud.map(\
      lambda k: process_s3_file(input_bucket_name, k, output_bucket_name), 
      matching_keys, _type = 'f2', _label='generate HDF' )
    
    progress = progressbar.ProgressBar(len(jids)).start()
    n_finished = 0
    for _ in cloud.iresult(jids):
      n_finished += 1
      progress.update(n_finished)
    progress.finish()
  else:
    print "Running locally..."
    print "%d keys match the pattern \'%s\'" % (len(matching_keys), key_glob)
    for key in matching_keys:
      process_s3_file(input_bucket_name, key, output_bucket_name)
  print "Done!"
  
parser = OptionParser(usage = "usage: %prog [options] path")
parser.add_option("-m", "--max-books", dest="max_books",
  type="int", help="maximum number of order books to read", default=None)
parser.add_option("-d", "--feature-dir",
  dest="feature_dir", default=None, type="string",
  help="which directory should we write feature files to")

parser.add_option('--heap-profile', default=False, action="store_true",
  help='Show heap statistics after exec (local only)')

if __name__ == '__main__':
  (options, args) = parser.parse_args()
  print "Args = ", args
  print "Options = ", options
  if len(args) != 1:
    parser.print_help()
  elif args[0].startswith('s3://'):
    bucket, _, pattern = args[0].split('s3://')[1].partition('/')
    assert bucket and len(bucket) > 0
    assert pattern and len(pattern) > 0
    print "Bucket = %s, pattern = %s" % (bucket, pattern)
    process_s3_files(bucket, pattern)
  else:
    process_local_dir(args[0],
      options.feature_dir,
      options.max_books, 
      options.heap_profile)