import numpy as np
import signals
import manual_classifiers
import os
import simulate2
import logging
import boto
import sys
from optparse import OptionParser

def test_strategy(strategy_name, hdf_filename, min_profit_prct=0.0003, signal_window_time=100, min_window_signals=1, trade_size_usd=1000000, max_position=5000000, fill_function=None):

    if hdf_filename is None:
        raise RuntimeError('HDF File must be specified') 
    if not os.path.exists(hdf_filename):
        raise RuntimeError('Cannot find path specifed')


    (d, currency_pair, ts, bids, offers, bid_vols, offer_vols) =  simulate2.load_dataset(hdf_filename)  

    #if strategy_name == "momentum1":
    (signals, mean_spread, mean_range) = strategy_name(d)
    #if strategy_name == "momentum2":
    #    (signals, mean_spread, mean_range) = manual_classifiers.momentum2(d)
    #if strategy_name == "active1":
    #    (signals, mean_spread, mean_range) = manual_classifiers.active1(d)

    (usd_pnl, pos_deltas, pos_run, closing_position, closing_pnl, usd_last_pos, ignored_signals) = \
            simulate2.execute_aggressive(ts, \
                                        bids, \
                                        offers, \
                                        bid_vols, \
                                        offer_vols, \
                                        signals, \
                                        currency_pair, \
                                        trade_size_usd, \
                                        signal_window_time, \
                                        min_window_signals, \
                                        min_profit_prct, \
                                        carry_position = False, \
                                        max_position = max_position, \
                                        fill_function=None, \
                                        cut_long = -(mean_spread+mean_range), \
                                        cut_short= -(mean_spread+mean_range))
    
    print "Min_profit_prct: ", min_profit_prct
    print "Signal_window_time: ", signal_window_time
    print "Min_window_signals: ", min_window_signals
    print "Trade_size_usd: ", trade_size_usd
    

    simulate2.trade_stats(d, \
                        strategy_name,\
                        ts,\
                        signals,\
                        min_profit_prct,\
                        signal_window_time,\
                        min_window_signals,\
                        trade_size_usd,\
                        max_position,\
                        usd_pnl,\
                        pos_deltas,\
                        closing_position,\
                        closing_pnl,\
                        mean_spread,\
                        mean_range,\
                        ignored_signals,\
                        tick_file = hdf_filename, \
                        out_file=hdf_filename + "." + "run" + ".txt") 
        

TEST_STRATEGIES=[manual_classifiers.momentum1, 
                manual_classifiers.momentum2,
                manual_classifiers.active1]

TEST_PARAMS = [
{'profit': 0.0003,
'signal_window': 300,
'min_signal_count': 9}
,
{'profit': 0.0001,
'signal_window': 100,
'min_signal_count': 1}
,
{'profit': 0.0002,
'signal_window': 100,
'min_signal_count': 1}
,
{'profit': 0.0003,
'signal_window': 300,
'min_signal_count': 2}
,
{'profit': 0.0003,
'signal_window': 100,
'min_signal_count': 1}
,
{'profit': 0.0004,
'signal_window': 100,
'min_signal_count': 1}
,
{'profit': 0.0004,
'signal_window': 300,
'min_signal_count': 2}
,
{'profit': 0.0005,
'signal_window': 100,
'min_signal_count': 1}
,
{'profit': 0.0007,
'signal_window': 100,
'min_signal_count': 1}
,
{'profit': 0.0007,
'signal_window': 500,
'min_signal_count': 3}
,
{'profit': 0.0009,
'signal_window': 300,
'min_signal_count': 2}
]

USE_EPHEMERAL = False
if USE_EPHEMERAL:
    EPHEMERAL0='/media/ephemeral0'
else:
    EPHEMERAL0='/home/timir/test_data/'

if __name__ == '__main__':
    sys.path.append('../aws_pipeline/')
    import s3_download_file
    parser = OptionParser()
    parser.add_option("-b", "--bucket", dest="bucket_name", help="Bucket name")
    (options, args) = parser.parse_args()
    if options.bucket_name is None:
        print "Must specify bucket name"
        sys.exit(-1)

    logging.basicConfig()
    s3cxn = boto.connect_s3()

    bucket = s3cxn.get_bucket(options.bucket_name)
    if len(args) == 0:
        keys = bucket.get_all_keys()
        keys = [k.name for k in keys]
    else:
        keys = args

    for k in keys:
        if k[-4:] == '.hdf':
            logging.debug("Downloading %s", k)
            if os.path.isfile(EPHEMERAL0 + k):
                print "Skipping download - file exists ", k
            else:
                s3_download_file.get_s3_file_to_local(s3cxn, bucket, k, k, EPHEMERAL0)
            for params in TEST_PARAMS:
                for strategy in TEST_STRATEGIES:
                    strategy_name = strategy
                    test_strategy(strategy_name, EPHEMERAL0+k, min_profit_prct=params['profit'], signal_window_time=params['signal_window'], min_window_signals=params['min_signal_count']) 



