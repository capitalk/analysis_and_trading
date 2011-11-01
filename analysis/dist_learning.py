
import numpy as np     
import cloud

from aws_helpers import make_s3_filenames, print_s3_hdf_files 
from dataset_helpers import load_s3_data
from features import features 
from evaluation import eval_prediction, eval_all_thresholds
import signals     
from encoder import FeatureEncoder
from treelearn import BaggedClassifier, SVM_Tree 

def get_dict(dicts, key):
    if key in dicts: return dicts[key]
    else: return {}


def loader(files):
    return load_s3_data(files, features=features, signal_fn=signals.bid_offer_cross) 
        
# load each file, extract features, concat them together 
def worker(params, features, train_files, test_files): 
    general_params = get_dict(params, 'general')
    train_params  = get_dict(params, 'training')
    encoder = params['encoder']
    model = params['model']
    
    
    print "General params:", general_params 
    print "Encoder:", encoder
    print "Model:", model
    print "Train params:", train_params     
    
    print "Loading training data..."
    
    train_data, train_signal, train_times, train_bids, train_offers, currencies = loader(train_files)
    ntrain = train_data.shape[0] 
    if 'target' in general_params: target = general_params['target']
    else: target = None
    if target: train_signal = (train_signal == target).astype('int')
            
    # assume all files from same currency pair 
    ccy = currencies[0]
    
    
    print "Encoding training data..." 
    train_data = encoder.fit_transform(train_data)
    print "Encoded shape:", train_data.shape 
    print "train_data[500]", train_data[500, :] 
    
    if 'class_weight' in train_params: 
        model.fit(train_data, train_signal, class_weight=train_params['class_weight'])
    else: 
        model.fit(train_data, train_signal)
    
    del train_data
    del train_signal 
    
    print "Reminder, here were the params:", params 
    print "Loading testing data..."
    test_data, test_signal, test_times, test_bids, test_offers, _ = loader(test_files)
    
    print "Encoding test data" 
    test_data = encoder.transform(test_data, in_place=True)
    
    print "test_data[500] =", test_data[500, :]
    print "Evaluating full model"
    
    prob = model.predict_proba(test_data, return_probs=True) 
    pred =  np.argmax(prob, axis=1)
    
    if target:
        test_signal = target * (test_signal == target).astype('int')
        target_index = model.classes.index(1)
        target_probs = probs[:,target_index]
        result = eval_all_thresholds(test_times, test_bids, test_offers, target, target_probs, test_signal, ccy)
    else: result = eval_prediction(test_times, test_bids, test_offers, pred, test_signal, ccy)
    
    print features
    print '[model]'
    print model
    print '[encoder]'
    print encoder.mean_
    print encoder.std_
    print '[result]' 
    print "precisions:", result['all_precisions']
    print "recalls:",  result['all_recalls']
    print 'threshold:', result['best_thresh']
    print 'precision:', result['best_precision']
    print 'recall:', result['best_recall']
    print 'f-score:', result['best_score']
    print 'ntrades:', result['ntrades']
    print 'ppt:', result['ppt']
    # have to clear sample weights since SGDClassifier stupidly keeps them 
    # after training 
    #model.sample_weight = [] 
    return {'params':params, 'result': result, 'encoder': encoder, 'model': model}
    
def cartesian_product(options):
    import itertools
    combinations = [x for x in apply(itertools.product, options.values())]
    return [dict(zip(options.keys(), p)) for p in combinations]

def prune(dicts, condition):
    result = []
    for d in dicts:
        if not condition(d):
            result.append(d)
    return result 
    
def gen_work_list(): 
    targets = [-1]
    oversampling_factors = [0]    
    
    class_weights = [15] #,2,3,4] 
    
    alphas = [0.000001] #, 0.0001, 0.01]

    possible_encoder_params = {
        'dictionary_type': [None, 'kmeans'],
        'dictionary_size': [None, 75],
        'pca_type': [None, 'whiten'], 
        'compute_pairwise_products': [False], 
        'binning': [False]
    }
    all_encoders = [
        FeatureEncoder(**p) 
        for p in cartesian_product(possible_encoder_params) 
        if (p['dictionary_type'] is not None or p['dictionary_size'] is None)
    ]
    
    possible_model_params = { 
        'C' : [0.1, 10.]
    }
    all_base_classifiers = [
        SVM_Tree(**params) 
        for params in cartesian_product(possible_model_params)
    ]
    
    possible_ensemble_params = {
        'base_classifier': all_base_classifiers,
        'num_classifiers': [25], #[100, 200]
        'weighting':  [0.25], 
        'stacking': [True, False], 
        'verbose': [True], 
        'sample_percent': [0.1],
    }
    all_ensembles = [
        BaggedClassifier(**params)
        for params in cartesian_product(possible_ensemble_params)
    ]
    
    worklist = [] 
    for target in targets:
        for smote_factor in oversampling_factors:
            general_params = {
                'oversampling_factor': smote_factor, 
                'target': target
            }
            for encoder in all_encoders:
                for model in all_ensembles:
                    for cw in class_weights:    
                        train_params = { 'class_weight': {0:1, 1:cw} }
                        params =  {
                            'general': general_params, 
                            'encoder': encoder, 
                            'model': model, 
                            'training': train_params
                        }
                        worklist.append (params)
    return worklist 

def init_cloud(): 
    cloud.config.force_serialize_debugging = False
    cloud.config.force_serialize_logging = False 
    cloud.config.commit()
    cloud.setkey(2579, "f228c0325cf687779264a0b0698b0cfe40148d65")

def param_search(train_files, test_files, features=features, debug=False):
    print "Features:", features 
    print "Training files:", train_files
    print "Testing files:", test_files 
    
    def eval_param(p): 
        return worker(p, features, train_files, test_files)
    
    params = gen_work_list()
    if debug: 
        print "[Debug mode]"
        result_list = map(eval_param, params[:1])
        for params, features, e, svm, result in result_list:
            print params, "=>", result 
    else: 
        init_cloud() 
        label = ", ".join(train_files)
        jobids = cloud.map(eval_param, params, _fast_serialization=2, _type='m1', _label=label, _env='param_search') 
        results = [] 
        print "Launched", len(params), "jobs, waiting for results..."
        for x in cloud.iresult(jobids):
            if x is not None:
                results.append(x)
                print x['params']
                print x['model']
                r = x['result']
                print 'Result:  precision =', r['best_precision'], 'recall =', r['best_recall'], 'ppt =', r['ppt'], 'ntrades =', r['ntrades']
                print "---" 
                
        def cmp(x,y):
            return int(np.sign(x['result']['best_score'] - y['result']['best_score']))
        
        results.sort(cmp=cmp)
        
        #accs = [x['result']['accuracy'] for x in results]
        #ppts = [x['result']['ppt'] for x in results]
        #print ppts 
        #print accs
        
        print "Best:"
        for item in results[-5:]:
            print item['params']
            r = item['result']
            print [(k, r[k]) for k in sorted(r.keys())]


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecn", dest="ecn", default="fxcm")
    parser.add_argument("--ccy", dest="ccy", help="currency pair (e.g. USDJPY)")
    parser.add_argument("--train", dest="train", help="training dates", nargs='*', default=[])
    parser.add_argument("--train_files", dest="train_files", help="training files", nargs='*', default=[])
    
    parser.add_argument("--test", dest="test", help="testing dates", nargs='*', default=[])
    parser.add_argument("--test_files", dest="test_files", help="testing files", nargs='*', default=[])
    parser.add_argument("--debug", action='store_true', default=False, dest='debug')
    #todo: make this actually work 
    parser.add_argument("--output", dest="output", default=None, help="output file for model and encoder")
    # todo: allow either parameter sweep or manually specify learning params  ie --thresh 0.9 0.95 
    args = parser.parse_args()
    if args.train == [] or args.test == []: print_s3_hdf_files()
    else: 
        training_files = make_s3_filenames(args.ecn, args.ccy, args.train) + args.train_files
        testing_files = make_s3_filenames(args.ecn, args.ccy, args.test) + args.test_files 
        param_search(training_files, testing_files, debug=args.debug) 
