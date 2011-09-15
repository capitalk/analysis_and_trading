#include <vector>
#include <string>
#include "predictor.h" 


int main(int argc, char* argv[]) {
    // pickled predictor copied from python 
    
    std::string pickle_str = "(ionline_predictor\nOnlinePredictor\np0\n(dp2\nS'min_frames_before_prediction'\np3\nI2\nsS'window_sizes'\np4\n(dp5\nS'50s'\np6\nI500\nsS'5s'\np7\nI50\nssS'encoder'\np8\n(itest_online_predictor\nDumbEncoder\np9\n(dp10\nbsS'raw_features'\np11\n(dp12\nS'bid'\np13\nconline_features\nbest_bid\np14\nsS'offer'\np15\nconline_features\nbest_offer\np16\nssS'feature_exprs'\np17\n(lp18\nS'(bid/mean/5s + offer/mean/5s) % 2'\np19\naS'(bid/mean/5s - bid/mean/50s) % bid/std/50s'\np20\nasS'model'\np21\n(itest_online_predictor\nDumbModel\np22\n(dp23\nbsS'longest_window'\np24\ncnumpy.core.multiarray\nscalar\np25\n(cnumpy\ndtype\np26\n(S'i8'\np27\nI0\nI1\ntp28\nRp29\n(I3\nS'<'\np30\nNNNI-1\nI-1\nI0\ntp31\nbS'\\xf4\\x01\\x00\\x00\\x00\\x00\\x00\\x00'\np32\ntp33\nRp34\nsS'feature_symbols'\np35\nc__builtin__\nset\np36\n((lp37\nS'bid/std/50s'\np38\naS'bid/mean/5s'\np39\naS'bid/mean/50s'\np40\naS'offer/mean/5s'\np41\natp42\nRp43\nsb.";
    std::cout << "Constructing predictor..."  << std::endl; 
    PythonPredictor p = PythonPredictor(pickle_str); 
    std::vector<float> bids; 
    bids.push_back(1.0f); 
    bids.push_back(0.99f); 
    
    std::vector<uint32_t> bid_sizes; 
    bid_sizes.push_back(2000); 
    bid_sizes.push_back(2500); 
    
    
    std::vector<float> offers; 
    offers.push_back(1.01f); 
    offers.push_back(1.02f); 
    
    std::vector<uint32_t> offer_sizes; 
    offer_sizes.push_back(1500); 
    offer_sizes.push_back(2500); 
    
    std::cout << "Calling update #1" << std::endl; 
    p.update(20, bids, bid_sizes, offers, offer_sizes); 
    std::cout << "Calling update #2" << std::endl; 
    p.update(30, bids, bid_sizes, offers, offer_sizes); 
    std::cout << "Calling predict #1" << std::endl; 
    p.predict(90); 
    bids[0] = 1.01f; 
    offers[0] = 1.015f; 
    std::cout << "Calling update #3" << std::endl; 
    p.update(120, bids, bid_sizes, offers, offer_sizes); 
    std::cout << "Calling predict #2" << std::endl; 
    int result = p.predict(190); 
    return 0; 
}
