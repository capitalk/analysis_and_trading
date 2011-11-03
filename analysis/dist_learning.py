
import numpy as np     
import cloud

from aws_helpers import make_s3_filenames, print_s3_hdf_files 
from dataset_helpers import load_s3_data
from features import features 
from evaluation import eval_prediction, eval_all_thresholds
import signals     
from encoder import FeatureEncoder
from treelearn import ClassifierEnsemble, RegressionEnsemble, ObliqueTree
from treelearn import mk_sgd_tree, mk_svm_tree 
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso 

def get_dict(dicts, key):
    if key in dicts: return dicts[key]
    else: return {}


def loader(files):
    return load_s3_data(files, features=features, signal_fn=) 
        
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
    
    probs = model.predict_proba(test_data) 
    pred =  np.argmax(probs, axis=1)
    
    if target:
        test_signal = target * (test_signal == target).astype('int')
        classes = list(model.classes)
        target_index = classes.index(1)
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
    
def init_cloud(): 
    cloud.config.force_serialize_debugging = False
    cloud.config.force_serialize_logging = False 
    cloud.config.commit()
    cloud.setkey(2579, "f228c0325cf687779264a0b0698b0cfe40148d65")

def param_search(
        signal = signals.bid_offer_cross, 
        ensemble = ClassifierEnsemble, 
        base_models=[mk_sgd_tree()],
        num_models = [50], 
        bagging_percents = [0.25], 
        dict_types = [None, 'kmeans'], 
        dict_sizes = [None, 50], 
        pca_types =  [None, 'whiten'], 
        compute_pairwise_products = [False], 
        binning = [False], 
        stacking_models = [None, LogisticRegression()],
        debug=False):
    print "Features:", features 
    print "Training files:", train_files
    print "Testing files:", test_files 
    
    def do_work(p): 
        return worker(p, features, train_files, test_files)
    
     targets = [-1]
    oversampling_factors = [0]    

    class_weights = ['auto'] 
    
    
    possible_encoder_params = {
        'dictionary_type': dict_types,
        'dictionary_size': dict_sizes, 
        'pca_type': pca_types, 
        'compute_pairwise_products': compute_pairwise_products, 
        'binning': binning, 
    }
    
    all_encoders = [
        FeatureEncoder(**p) 
        for p in cartesian_product(possible_encoder_params) 
        if (p['dictionary_size'] is not  None or p['dictionary_type'] is None)
    ]
    
    possible_ensemble_params = {
        'base_model': base_models,
        'num_models': num_models, 
        'weighting':  [0.25], 
        'stacking_model': stacking_models, 
        'verbose': [True], 
        'bagging_percent': bagging_percents,
    }
    all_ensembles = [
        ensemble(**params)
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
                        train_params = { 'class_weight': cw }
                        params =  {
                            'general': general_params, 
                            'encoder': encoder, 
                            'model': model, 
                            'training': train_params
                        }
                        worklist.append (params)
    if debug: 
        print "[Debug mode]"
        result_list = map(do_work, worklist[:1])
        for params, features, e, svm, result in result_list:
            print params, "=>", result 
    else: 
        init_cloud() 
        label = ", ".join(train_files)
        jobids = cloud.map(do_work, worklist, _fast_serialization=2, _type='m1', _label=label, _env='param_search') 
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
    
    parser.add_argument("--output", dest="output", default=None, help="output file for model and encoder")
    
    
    parser.add_argument("--dict_size", dest="dict_size", nargs="*", default=[None, 'kmeans'])
    parser.add_argument("--dict_type", dest="dict_type", nargs="*", default=[None, 50]), 
    parser.add_argument("--bagging_prct", dest="bagging_prct", nargs="*", default=[0.25]), 
    parser.add_argument("--regression", dest='regression', action='store_true', default=False)
    
    
    args = parser.parse_args()
    if args.train == [] or args.test == []: print_s3_hdf_files()
    else: 
        training_files = make_s3_filenames(args.ecn, args.ccy, args.train) + args.train_files
        testing_files = make_s3_filenames(args.ecn, args.ccy, args.test) + args.test_files 
        
        if args.regression: 
            ensemble = RegressionEnsemble
            base_models = [mk_regression_tree()] 
            stacking_models = [None, LinearRegression]
            signals = signals.
        else:
            ensemble = ClassifierEnsemble
            base_models =[mk_sgd_tree()]
            stacking_models = [None, LogisticRegression]
            signal = signals.bid_offer_cross
            
        param_search(training_files, testing_files, debug=args.debug) 
