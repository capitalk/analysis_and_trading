# min_profit_pct is used for two checks
# 1) on signal check if enough profit to exit - otherwise ignore the signal
# 2) trading opportunisitically - when trade_only_on_signal = False we check every timestamp for a profitable exit as defined by min_profit_pct
# TODO - should change trade_only_on_signal to be take_profit_pct and set that to a num or None 
# Set min_profit_pct to 0 to always trade on signal regardless of profit
# Set min_profit_pct to 100 to "never" trade take profit when trade_only_on_signal = False
#
import numpy as np
import simulate2
currency_pair = "CC1/CC2"

def test_TRADE_AGGRESSIVE():
    long_vol = 0;
    short_vol = 0;
    long_paid = 0;
    short_recv = 0;
    this_trade_pnl = 0;

    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(0, 0, 0, 0, "B", 1, 1.0, 1.01, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 1 and short_vol == 0 and short_recv == 0 and long_paid == 1.01 and this_trade_pnl == 0)

    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B", 1, 1.02, 1.03, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 2 and short_vol == 0 and short_recv == 0 and long_paid == 2.04 and this_trade_pnl == 0)

    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", 3, 1.03, 1.04, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 2 and short_vol == 3 and short_recv == 3.09 and long_paid == 2.04 and round(this_trade_pnl, 2) == 0.02)

    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "B",1 , 1.03, 1.04, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 3 and short_vol == 3 and short_recv == 3.09 and long_paid == 3.08 and round(this_trade_pnl, 2) == -0.01)

    long_vol = short_vol = long_paid = short_recv = this_trade_pnl = 0.0 
    # Try trading through the book
    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(0, 0, 0, 0, "B", 11, 1.0, 1.01, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 11 and short_vol == 0 and short_recv == 0 and round(long_paid,5) == 11.11055 and this_trade_pnl == 0)

    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", 11, 1.02, 1.03, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 11 and short_vol == 11 and round(short_recv,5) == 11.21945 and round(long_paid,5) == 11.11055 and round(this_trade_pnl, 2) == 0.11)

    # Try trading through the book
    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(0, 0, 0, 0, "B", 11, 1.0, 1.01, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 11 and short_vol == 0 and short_recv == 0 and round(long_paid,5) == 11.11055 and this_trade_pnl == 0)

    (long_vol, short_vol, long_paid, short_recv, this_trade_pnl) = simulate2.TRADE_AGGRESSIVE(long_vol, short_vol, long_paid, short_recv, "S", 11, 1.02, 1.03, 10, 10, 1)
    print "long_vol: ", long_vol, " short_vol: ", short_vol, " short_recv: ", short_recv, " long_paid: ", long_paid, " this_trade_pnl: ", this_trade_pnl
    assert(long_vol == 11 and short_vol == 11 and round(short_recv,5) == 11.21945 and round(long_paid,5) == 11.11055 and round(this_trade_pnl, 2) == 0.11)

ts = np.arange(1,11)
#################################################################################
# Long trade tests
#################################################################################
def test_long1():
    # Set up the market
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0.1
    mean_range = 0.0
    # Set up signals for single long trade
    signals  = [1,0,0,0,0,0,0,0,0,-1]

    # TEST 1
    # Test enter and exit wiht one buy and sell - no position carry, trade only on signal = True
    # Start with long and end with short
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=0.0001, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=None)

    assert(round(sum(pnl),2) == 0.80)
    assert(sum(position_deltas) == 0.0)
    assert(sum(position_running) == 9.)
    assert(closing_position == 0)

def long_test_2():
    # Set up the market
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0.1
    mean_range = 0.0
    # Set up signals for single long trade
    signals  = [1,0,0,0,0,0,0,0,0,-1]
    # TEST 2
    # Test enter and exit with one buy and sell - no position carry, trade only on signal = False
    # Test for opportunistic profit
    # NB - NO CARRY will average closing prices which again distorts pnl in monotonic markets like the test sets so closing pnl will be exaggerated here
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=0.0001, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=0.0001)
    # Profit here is convoluted since the price array is artificial and changes very quickly so when averaging the price to close the position the true price is distorted - i.e. first 4 offers average to 1.30
    assert(round(sum(pnl),2) == 0.35)

    pos_deltas_test = [ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0., 1.]
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)
    assert(sum(position_deltas) + closing_position == 0)

    pos_run_test  = [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)
    

    assert(round(closing_pnl,2) == 0.25)
    assert(closing_position == -1.)

def long_test3():
    # Set up the market
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0.1
    mean_range = 0.0
    # Set up signals for single long trade
    signals  = [1,0,0,0,0,0,0,0,0,-1]
    # TEST 3
    # Test for evaluating profit prct - this shoudl never trigger take profit
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=100, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=0.0001)
    # Profit here is convoluted since the price array is artificial and changes very quickly so when averaging the price to close the position the true price is distorted - i.e. first 4 offers average to 1.30
    assert(round(sum(pnl),2) == 0.35)

    pos_deltas_test = [ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0., 1.]
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test  = [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

#################################################################################
# Short trade tests
#################################################################################
def short_test1():
    # Set up the market
    bids = np.arange(2, 1, -.10)
    offers = np.arange(2.1, 1.1, -.10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0.1
    mean_range = 0.0
    # Start with short and end with long
    signals = [-1,0,0,0,0,0,0,0,0,1]

    # TEST 4
    # Test short trade first - no carry position, trade only on signal = True
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=0.0001, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=None)
    assert(round(sum(pnl), 2) == 0.8)

    pos_deltas_test = [ -1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 1.]
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test  = [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  0.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(round(closing_pnl,2) == 0.0)
    assert(round(closing_position,2) == 0.0)

def short_test2():
    # Set up the market
    bids = np.arange(2, 1, -.10)
    offers = np.arange(2.1, 1.1, -.10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0.1
    mean_range = 0.0
    # Start with short and end with long
    signals = [-1,0,0,0,0,0,0,0,0,1]
    # TEST 5
    # Test short trade first - no carry position, trade only on signal = False
    # Short and take profit - then trade last frame long and close position with short (no carry)
    # NB - NO CARRY will average closing prices which again distorts pnl in monotonic markets like the test sets so closing pnl will be exaggerated here
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=0.0001, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=0.0001)
    assert(round(sum(pnl), 2) == 0.35)

    pos_deltas_test = [ -1.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  0., -1.]
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test  = [ -1.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(round(closing_pnl,2) == 0.25)
    assert(closing_position == 1.)

def short_test3():
    # Set up the market
    # Cut level is offer[6]
    offers = [ 2.1,  2. ,  1.9,  1.8,  1.7,  1.6,  2.5 ,  1.4,  1.3,  1.2]
    bids = [ 2. ,  1.9,  1.8,  1.7,  1.6,  1.5,  1.4,  1.3,  1.2,  1.1]
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0.1
    mean_range = 0.0
    # Start with short and end with long
    signals = [-1,0,0,0,0,0,0,0,0,1]
    # TEST 6
    # Test short trade with cutoff
    # NB - NO CARRY will average closing prices which again distorts pnl in monotonic markets like the test sets so closing pnl will be exaggerated here
    mean_spread = 0.0
    mean_range = 0.0001
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=0.0001, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=None)
    assert(round(sum(pnl), 2) == -0.55)

    pos_deltas_test = [-1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  -1.]
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test  = [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  0.,  0.,  0.,  0.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(round(closing_pnl,2) == -0.05)
    assert(closing_position == 1.)

#################################################################################
# Mixed trade tests
#################################################################################

def mix_test1():
    signals  = [1,1,1,0,-1,0,-1,0,1,-1]
    mean_spread = 1
    mean_range = 1
    # Reset the market 
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)

    # TEST 7
    # No take profit - trade only on signals 
    # No cuts - set spread and range high 
    # DO carry position - elimintates averaging problem of closing trde - pnl more transparent
    # No min profit - trade on all signals - not just profitable ones
    # This should trade on ALL signals regardless of profitability and never cut 
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=None, carry_position = True, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=None)

    assert(round(sum(pnl), 2) == .15)

    pos_deltas_test = [ 1.,  1.,  1.,  0., -3.,  0., -1.,  0.,  1., -1.] 
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    # End with position since carry_position is True here
    pos_run_test  = [ 1.,  2.,  3.,  3.,  0.,  0., -1.,  -1.,  0., -1.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(closing_position == -1.)
    assert(closing_pnl == 0)


def mix_test2():
    signals  = [1,1,1,0,-1,0,-1,0,1,-1]
    mean_spread = 1
    mean_range = 1
    # Reset the market 
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    # TEST 8
    # No take profit - trade only on signals 
    # No cuts - set spread and range high 
    # DO NOT carry position - close avg price of long/short against avg of past market
    # No min profit - trade on all signals - not just profitable ones
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=None, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=None)

    assert(round(sum(pnl), 2) == .39)

    pos_deltas_test = [ 1.,  1.,  1.,  0., -3.,  0., -1.,  0.,  1., 1.]
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test  = [ 1.,  2.,  3.,  3.,  0.,  0., -1.,  -1.,  0.,  0.] 
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(closing_position == -1.)
    assert(round(closing_pnl, 2) == 0.24)

def mix_test3():
    signals  = [1,1,1,0,-1,0,-1,0,1,-1]
    # Reset the market 
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0
    mean_range = 0.0001
    # TEST 9
    # No take profit - trade only on signals 
    # Allow cuts 
    # DO NOT carry position - close avg price of long/short against avg of past market
    # No min profit - trade on all signals - not just profitable ones
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=None, carry_position = False, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=None)

    assert(round(sum(pnl), 2) == 0.69)

    pos_deltas_test = [ 1.,  1.,  1.,  0., -3.,  0., -1.,  1.,  1., -1.] 
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test  = [ 1.,  2.,  3.,  3.,  0.,  0., -1.,  0.,  1.,  0.]
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(closing_position == 0.)
    assert(round(closing_pnl, 2) == 0.)

def mix_test4():
    signals  = [1,1,1,0,-1,0,-1,0,1,-1]
    # Reset the market 
    bids = np.arange(1, 2, .10)
    offers = np.arange(1.1, 2.1, .10)
    bid_vols = 2000000 * np.ones(10)
    offer_vols = 1000000 * np.ones(10)
    mean_spread = 0
    mean_range = 0.0001
    # TEST 10
    # Allow take profit - fully exit profitable position
    # Allow cuts 
    mean_spread = 0
    mean_range = 0.0001
    # DO NOT carry position - close avg price of long/short against avg of past market
    # No min profit - trade on all signals - not just profitable ones
    (pnl, position_deltas, position_running, closing_position, closing_pnl, ignored_signals, m2m_pnl) = simulate2.execute_aggressive(ts, bids, offers, bid_vols, offer_vols, signals, currency_pair, signal_window_time=1, min_window_signals=1, min_profit_prct=None, carry_position = True, default_trade_size = 1, max_position=5, fill_function=None, cut_long = -(mean_spread+mean_range)*2, cut_short= -(mean_spread+mean_range)*2, usd_transaction_cost= 0, trade_file='/tmp/testoutshit', take_profit_pct=0.0001)

    assert(round(sum(pnl), 2) == 0.02)

    pos_deltas_test = [ 1.,  1.,  1., -3., -1.,  1., -1.,  1.,  1., -1.] 
    pos_deltas_out = np.equal(position_deltas, pos_deltas_test)
    assert(pos_deltas_out.all() == True)

    pos_run_test = [ 1.,  2.,  3.,  0., -1.,  0., -1.,  0.,  1.,  0.] 
    pos_run_out = np.equal(position_running, pos_run_test)
    assert(pos_run_out.all() == True)

    assert(closing_position == 0.)
    assert(closing_pnl == 0.)



