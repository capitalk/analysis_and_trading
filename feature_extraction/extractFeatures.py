

import numpy as np  
import scipy.stats
import pylab 

import buildBook

from features import * 
from aggregators import approx_skewness, ols, ols_1000x, mean_crossing_rate
import infer_actions
from featurePipeline import FeaturePipeline
from optparse import *
import os, os.path
import cloud, h5py, gzip, gc, datetime

parser = OptionParser(usage = "usage: %prog [options] path")
parser.add_option("-m", "--max_books", dest="max_books", type="int",
                  help="maximum number of order books to read", default=None)
parser.add_option("-d", "--feature_dir",
                  dest="feature_dir", 
                  default=None, 
                  type="string",
                  help="which directory should we write feature files to")
parser.add_option("-p", "--profile", dest="profile", action="store_true", default=False, help="run inside profiler")
parser.add_option("-c", "--cloud", dest="use_cloud", action="store_true", default=False, help="run computations using picloud")
parser.add_option("-s", "--cloud_sim", dest="cloud_sim", action="store_true", default=False, help="use cloud simulator")
parser.add_option("-a", "--aggregate", dest="aggregate", action="store_true", default=False, help="aggregate features over longer time scales")
parser.add_option("--heap_profile", dest="heap_profile", action='store_true', default=False, help="print information about live heap objects")
parser.add_option("-n", "--num_processors", dest="num_processors", type="int", default=0, help="number of processors when not using cloud")

(options, args) = parser.parse_args()
print "Args = ", args
print "Options = ", options

timescales = [("1s", 1000), ("5s", 5000),  ("50s", 50000) ]
extractor = FeaturePipeline(timescales=timescales)
extractor.add_feature('t', millisecond_timestamp)
extractor.add_feature('bid', best_bid)
extractor.add_feature('offer', best_offer)
extractor.add_feature('bid_range', bid_range)
extractor.add_feature('offer_range', offer_range)
extractor.add_feature('spread', spread)
extractor.add_feature('midprice', midprice)
extractor.add_feature('weighted_total_price', volume_weighted_overall_price)
extractor.add_feature('offer_vol', best_offer_volume)
extractor.add_feature('bid_vol', best_bid_volume)
extractor.add_feature('total_bid_vol', bid_volume)
extractor.add_feature('total_offer_vol', offer_volume)
extractor.add_feature('t_mod_1000', fraction_of_second, use_window_reducers=False)
extractor.add_feature('message_count', message_count, sum_100ms=True)
# V3 orderbook action  features 
extractor.add_feature('added', total_added_volume, sum_100ms=True)
extractor.add_feature('deleted', total_deleted_volume, sum_100ms=True)
extractor.add_feature('net_action_volume', net_volume, sum_100ms=True)
extractor.add_feature('filled', fill_volume, sum_100ms=True)
extractor.add_feature('canceled', canceled_volume, sum_100ms=True)
extractor.add_feature('insertion_flow', insertion_flow)

if options.aggregate: 
    extractor.add_reducer('mean',np.mean)
    extractor.add_reducer('std', np.std)
    extractor.add_reducer('max', np.max)
    extractor.add_reducer('min', np.min)
    extractor.add_reducer('slope', ols_1000x, uses_time= True)
    extractor.add_reducer('mcr', mean_crossing_rate)


 # file exists and 'finished' flag is true 
def file_already_done(filename):
    if not os.path.exists(filename): return False 
    try:
        f = h5py.File(filename, 'r')
        finished = 'finished' in f.attrs and f.attrs['finished']
        f.close() 
        return finished
    except:
        return False

#def open_gzip_subprocess(filename): 
#    from subprocess import PIPE, Popen
#    p = Popen(["zcat", filename], stdout=PIPE)
#    return p.stdout

def open_gzip(filename):
    return gzip.GzipFile(filename, 'r')

def process_file_locally(input_filename, dest_filename): 
    print "Start time:", datetime.datetime.now()
    if filename.endswith('.gz'): 
        f = open_gzip(input_filename)
    else:
        f = open(input_filename, 'r')    
    
    if "JPY" in filename:
        extractor.add_feature('last_bid_digit_near_zero', second_bid_digit_close_to_wrap)
        extractor.add_feature('last_offer_digit_near_zero', second_offer_digit_close_to_wrap)
    else:
        extractor.add_feature('last_bid_digit_near_zero', fourth_bid_digit_close_to_wrap)
        extractor.add_feature('last_offer_digit_near_zero', fourth_offer_digit_close_to_wrap)
        
    if options.profile:
        import cProfile 
        cProfile.runctx("extractor.run(f, dest_filename, max_books = options.max_books)", globals(), locals(), filename="profile.cprof")
        import pstats
        stats = pstats.Stats("profile.cprof")
        stats.strip_dirs().sort_stats('time').print_stats(20)
    else:
        extractor.run(f, dest_filename, max_books = options.max_books)
    f.close()
    gc.collect()
    if options.heap_profile:
        print "Heap contents:"
        from guppy import hpy
        heap = hpy().heap()
        print heap
        print heap[0].rp
        print heap[0].byid
        
def process_file_worker(input_filename, output_filename): 
    short_input_name = os.path.basename(input_filename)
    cloud.files.get(input_filename, short_input_name)
    short_output_name = os.path.basename(output_filename)
    process_file_locally(short_input_name, short_output_name)
    cloud.files.put(short_output_name, name = output_filename)
    
def process_file_on_cloud(input_filename, output_filename): 
    if not cloud.files.exists(input_filename): 
        print "Putting", input_filename, "onto cloud.files"
        cloud.files.put(input_filename, name=input_filename)
    jobid = cloud.call(process_file_worker, input_filename, output_filename, _high_cpu=True, _label=input_filename, _fast_serialization=2)
    return jobid


if len(args) != 1:
    parser.print_help()
else:
    cloud.config.max_transmit_data = 56700000
    cloud.config.num_procs = options.num_processors
    if options.cloud_sim: 
        cloud.start_simulator()
    else:
        cloud.config.force_serialize_debugging = False
        cloud.config.force_serialize_logging = False 
    cloud.config.commit()
    if options.use_cloud: 
        cloud.setkey(2579, "f228c0325cf687779264a0b0698b0cfe40148d65")
        
    path = args[0]
    if not os.path.exists(path): 
        print "Specified path does not exist: ", path 
        exit()
    if os.path.isdir(path):
        files = os.listdir(path)
        basedir = path
    else:
        basedir = os.path.split(path)[0] 
        files = [os.path.basename(path)]
    
    if options.feature_dir:
        featureDir = options.feature_dir
    else:
        featureDir = os.path.join(basedir, 'features')
        
    if not os.path.exists(featureDir):
        os.makedirs(featureDir)
    
    count = 0
    cloud_jobs = [] 
    for filename in files:
        base = None
        if filename.endswith('.csv'):
            base = filename[0:-4]
        elif filename.endswith('.csv.gz'):
            base = filename[0:-7]
            
        if base is not None:
            count += 1
            input_filename = os.path.join(basedir, filename)
            print "----"
            print "Processing  #", count, " : ", input_filename 
            dest_filename = os.path.join(featureDir, base + ".hdf")                
            if file_already_done(dest_filename):
                print "Skipping, found data file", dest_filename
            elif options.use_cloud:
                print "Launching cloud job..." 
                jobid = process_file_on_cloud(input_filename, dest_filename)
                cloud_jobs.append((jobid, dest_filename))
            else:
                process_file_locally(input_filename, dest_filename)
                
    if len(cloud_jobs) > 0:
        print "Getting cloud results..." 
        
    for jobid, dest_filename in cloud_jobs:
        # block until done 
        cloud.join(jobid)
        print "Retrieving", dest_filename, "from cloud.files ", result 
        cloud.files.get(dest_filename, dest_filename)
        
