/*
 * Copyright (C) 2011 Capital K Partners BV
 * 
 * Be Kind - Rewind
 */ 

/*  
 *  PredictStrategy.cpp
 *  Strategies
 *
 *
 */

// TBricks
#include "tbricks_definitions.h" // Generated! DO NOT MODIFY

// CapK
#include "KSingleIVIDLeg.h"
#include "PredictStrategy.h"
#include "Utils.h"

// STD
#include <cmath>
#include <math.h>
#include <vector>

const tbricks::Price ONE_PIP = 0.0001;

const tbricks::String PLUGIN_NAME("Predict");

using namespace tbricks;

Predict::Predict(const tbricks::InitializationReason & reason, const tbricks::StrategyParameters & parameters) : 
m_initializationReason(reason),
m_instrumentParam(strategy_parameters::Instrument()),
m_portfolio(strategy_parameters::Portfolio()),
m_positionStream(*this),
m_sharedOrderManager(),
m_legTimer(*this)
{ 
	TBDEBUG(">Predict::Predict()");
	m_allSnapshotsComplete = false;

	// We accept all parameters as is
	// KTK - recheck this assumption
	UpdateParameters(parameters);
	SetName(PLUGIN_NAME);

	switch (reason.GetReason()) {
		case InitializationReason::CREATED:
			TBNOTICE("Created on user request");
			break;

		case InitializationReason::DUPLICATED:
			TBNOTICE("Created by cloning");
			break;
	
		case InitializationReason::RECOVERED:
			TBNOTICE("Recovered");
			// KTK - undelete next line when caught in crash loop so strategy auto deletes
			SetState(StrategyState::DELETED);
			//SetState(StrategyState::PAUSED);
			break;
	}

	TBWARNING("Clearing instrument list => +100ms");
	GetInstruments().Clear();

	if (m_sharedOrderManager.StartRecovery(reason)) {
			TBNOTICE("!!!!> TradeFrame - starting recovery");
	};

	SetName(PLUGIN_NAME);
	TBDEBUG("!!!!> Current state: " << GetState());
	TBDEBUG("<Predict::Predict()");
    
    std::string pickle_str = "(ionline_predictor\nOnlinePredictor\np0\n(dp2\nS'min_frames_before_prediction'\np3\nI2\nsS'window_sizes'\np4\n(dp5\nS'50s'\np6\nI500\nsS'5s'\np7\nI50\nssS'encoder'\np8\n(itest_online_predictor\nDumbEncoder\np9\n(dp10\nbsS'raw_features'\np11\n(dp12\nS'bid'\np13\nconline_features\nbest_bid\np14\nsS'offer'\np15\nconline_features\nbest_offer\np16\nssS'feature_exprs'\np17\n(lp18\nS'(bid/mean/5s + offer/mean/5s) % 2'\np19\naS'(bid/mean/5s - bid/mean/50s) % bid/std/50s'\np20\nasS'model'\np21\n(itest_online_predictor\nDumbModel\np22\n(dp23\nbsS'longest_window'\np24\ncnumpy.core.multiarray\nscalar\np25\n(cnumpy\ndtype\np26\n(S'i8'\np27\nI0\nI1\ntp28\nRp29\n(I3\nS'<'\np30\nNNNI-1\nI-1\nI0\ntp31\nbS'\\xf4\\x01\\x00\\x00\\x00\\x00\\x00\\x00'\np32\ntp33\nRp34\nsS'feature_symbols'\np35\nc__builtin__\nset\np36\n((lp37\nS'bid/std/50s'\np38\naS'bid/mean/5s'\np39\naS'bid/mean/50s'\np40\naS'offer/mean/5s'\np41\natp42\nRp43\nsb.";
    this->predictor = new PythonPredictor(pickle_str); 
}

void Predict::DestroyLegs(void)
{
	TBDEBUG("-Predict::DestroyLegs()");
	tbricks::Hash<tbricks::InstrumentVenueIdentification, KSingleIVIDLeg*>::iterator it; 
	for (it = m_ividToLeg.begin(); it != m_ividToLeg.end(); it++) {
		if (it->second != NULL) {
			it->second->CancelAll();
			delete (it->second);
		}
	}	
}

Predict::~Predict(void)
{
	TBDEBUG("-Predict::~Predict()");
	DestroyLegs();
	if (m_pBBOBook) {
		delete m_pBBOBook;
	}
    delete this->predictor; 
}

void Predict::HandleRunRequest(void)
{
	TBDEBUG(">Predict::HandleRunRequest()");
	if (GetState() == StrategyState::RUNNING) {
		TBWARNING("Already running state - IGNORING run request");
		return;
	}
	
	// Open the position stream just to have a double check on the position we are calculating
	m_positionStream.Open(Stream::SNAPSHOT_AND_LIVE, m_portfolio, tbricks::Position::AGGREGATE_POSITIONS_BY_CURRENCY);


	/* The instrument is the only piece of information we care about here 
 	 * since all streams will opened based on the instrument. We get the identifier and 
 	 * immediately turn it into an instrument which is then used to open the aggregated 
 	 * order book. 
 	 * Once the order book is openened and consistent then we begin recieving updates 
 	 * for prices and can begin looking for crossed markets
 	 */
	m_instrumentIdentifier = m_instrumentParam.GetInstrumentIdentifier();
	m_instrument = tbricks::Instrument(m_instrumentIdentifier);
	m_instruments.push_back(m_instrumentIdentifier);	
	SetInstruments(m_instruments);

	/* Shouldn't need value factor for FX - for futures or LSE equities */
	m_valueFactor = getValueFactor(m_instrumentIdentifier);
	TBSTATUS("ValueFactor: " << m_valueFactor);


	// Create the aggregated book which does nothing but set the instrument 
	// Opening the book opens all the streams
	m_pBBOBook = new KAggregatedOrderBook(m_instrument);
	if (m_pBBOBook) {
		m_pBBOBook->AddHandler(this);
		m_pBBOBook->Open(tbricks::Stream::SNAPSHOT_AND_LIVE, false);
	}

	// Get ticks sizes (in ITI) from KAggregatedOrderBook
	m_pBBOBook->getInstrumentTradingInformation(m_ITI); 	

	// Clean up old legs 
	// KTK - do we NEED to do this? 
	DestroyLegs();
	
	// Create legs for all venues of trade
	// KTK - note that we need to find the intersection of trading and market
	// ivids if market and trade are on different ivids
	m_tradingIvids = m_pBBOBook->getTradingIvids();	
	IVIDVector::iterator i;
	for (i = m_tradingIvids.begin(); i != m_tradingIvids.end(); i++) {
		TBDEBUG("Using MIC=> " << i->GetMIC());
		// Map this leg into structure		
		// Now each IVID maps to a running leg
		m_ividToLeg[(*i)] = new KSingleIVIDLeg(*this, *this, m_sharedOrderManager, *i, m_portfolio, "String");
	}

	// KTK - create legs for each IVID
/*
	m_pLeg1 = new SingleIVIDLeg(*this, *this, m_sharedOrderManager, m_ivid1, m_portfolio, m_instrument1.GetShortName());
	m_pLeg2 = new SingleIVIDLeg(*this, *this, m_sharedOrderManager, m_ivid2, m_portfolio, m_instrument2.GetShortName());
	m_pLeg3 = new SingleIVIDLeg(*this, *this, m_sharedOrderManager, m_ivid3, m_portfolio, m_instrument3.GetShortName());
*/

/*  For reference on getting tick rules, round lots and currency
	tbricks::TickRule tickRule(m_instrumentTradingInformation.GetTickRule());
	tbricks::Volume roundLot(m_instrumentTradingInformation.GetLotSize());
	tbricks::Currency currency(m_instrumentTradingInformation.GetCurrency());
*/

/*
	TBDEBUG("====> Opening streams");
	bool openCoalesced = false;
	m_bboStream1.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid1, AnyBestPriceFilter(), openCoalesced);
	m_bboStream2.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid2, AnyBestPriceFilter(), openCoalesced);
	m_bboStream3.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid3, AnyBestPriceFilter(), openCoalesced);

	m_bboStream1.SuppressVolumeUpdates(true);
	m_bboStream2.SuppressVolumeUpdates(true);
	m_bboStream3.SuppressVolumeUpdates(true);

	m_instStatusStream1.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid1, InstrumentByIdentifierFilter(m_instrumentIdentifier1));
	m_instStatusStream2.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid2, InstrumentByIdentifierFilter(m_instrumentIdentifier2));
	m_instStatusStream3.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid3, InstrumentByIdentifierFilter(m_instrumentIdentifier3));

	m_statsStream1.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid1); 
	m_statsStream2.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid2); 
	m_statsStream3.Open(Stream::SNAPSHOT_AND_LIVE, m_ivid3); 

	m_pLeg1->setActive(true);
	m_pLeg2->setActive(true);
	m_pLeg3->setActive(true);
*/

	TBDEBUG("====> Setting requested state: RUNNING");
	SetState(StrategyState::RUNNING);	
	
	TBDEBUG("<Predict::HandleRunRequest()");
}

void Predict::HandlePauseRequest(void)
{
	TBDEBUG(">Predict::HandlePauseRequest()");
	// KTK - deactivate all legs - either:
	// 1) pull all orders
	// 2) wait until delta flat and stop
	// We'll start with 1
	if (GetState() != StrategyState::PAUSED) {
		tbricks::Hash<tbricks::InstrumentVenueIdentification, KSingleIVIDLeg*>::iterator it;
		for (it = m_ividToLeg.begin(); it != m_ividToLeg.end(); it++) {
			// Setting active to false will cancel all orders
			it->second->SetActive(false);
		}
	}
	CloseAllStreams();
	//UpdateState();
	SetState(StrategyState::PAUSED);
	TBDEBUG("<Predict::HandlePauseRequest()");
}

void Predict::HandleDeleteRequest(void)
{
	TBDEBUG(">Predict::HandleDeleteRequest()");
	// KTK - deactivate all legs - either:
	// 1) pull all orders
	// 2) wait until delta flat and stop
	// We'll start with 1
	if (GetState() != StrategyState::DELETED) {
		tbricks::Hash<tbricks::InstrumentVenueIdentification, KSingleIVIDLeg*>::iterator it;
		for (it = m_ividToLeg.begin(); it != m_ividToLeg.end(); it++) {
			// Setting active to false will cancel all orders
			it->second->SetActive(false);
		}
	}
	CloseAllStreams();
	//UpdateState();
	SetState(StrategyState::DELETED);
	TBDEBUG("<Predict::HandleDeleteRequest()");
}

void Predict::CloseAllStreams(void)
{
	TBDEBUG(">Predict::CloseAllStreams()");
	m_pBBOBook->Close();
	TBDEBUG("<Predict::CloseAllStreams()");

}


/* Delete all active orders */
void Predict::DeleteAllOrders(void)
{
	TBDEBUG(">Predict::DeleteAllOrders()");
	if (GetState() != StrategyState::DELETED) {
		tbricks::Hash<tbricks::InstrumentVenueIdentification, KSingleIVIDLeg*>::iterator it;
		for (it = m_ividToLeg.begin(); it != m_ividToLeg.end(); it++) {
			// Setting active to false will cancel all orders
			it->second->SetActive(false);
		}
	}
/*
	
	if (GetState() != StrategyState::DELETED) {
		RawOrderHashIterator it = m_rawOrders.begin();
		while (it != m_rawOrders.end()) {
			Identifier request_id = SendDeleteRequest(it->first, *this);
			TBDEBUG("====> DELETE ORDER: " << it->first);
			m_pendingRequests[request_id] = tbricks::TransactionState::PENDING;
			it++;
		}
	}
*/
	TBDEBUG("<Predict::DeleteAllOrders() - DONE");
}

/* Manage requests from client to change parameters or attributes of strategy 
 * should set state where needed or SetTransactionPending(StrategyTransactionOperation) -
 * i.e. SetTransactionPending(STRATEGY_RUN) which would indicate that the strategy is pending 
 * RUN once the modify request is processed 
 */
void Predict::HandleModifyRequest(const tbricks::StrategyModifier & modifier)
{
	TBDEBUG(">Predict::HandleModifyRequest()");// << modifier);
        // Don't allow parameter changes once the price streams are aopen 
        // Should not allow changes to instrucments but allow changes to the other parameters
        // Also - should check if strategy is running rather than checking if pricestream is open
	if (GetState() == tbricks::StrategyState::RUNNING) {
		TBDEBUG("====> Modify not allowed when running.");
		return;
	}
	const StrategyParameters & modifiedParameters = modifier.GetParameters();
	UpdateParameters(modifiedParameters);

	TBDEBUG("<Predict::HandleModifyRequest()");
}



// Called when strategy panel is modified fron X-Ray - in strategy layout panel ONLY (not the one in the developer section)
// but rather the one under the instruments (the one you get when you double click on an instrument)
void Predict::HandleValidateRequest(tbricks::ValidationContext &context)
{
	TBDEBUG(">Predict::HandleValidateRequest()"); //  << context);
	// Don't need to do any validation since there is nothing selectable by the user
/*
	VenueValidation(context, m_instrumentParam1.GetDefinition(), m_venue1.GetDefinition());
	VenueValidation(context, m_instrumentParam2.GetDefinition(), m_venue2.GetDefinition());
	VenueValidation(context, m_instrumentParam3.GetDefinition(), m_venue3.GetDefinition(), false);

	MICValidation(context, m_instrumentParam1.GetDefinition(), m_venue1.GetDefinition(), m_mic1.GetDefinition(), true);
	MICValidation(context, m_instrumentParam1.GetDefinition(), m_venue2.GetDefinition(), m_mic2.GetDefinition(), true);
	MICValidation(context, m_instrumentParam1.GetDefinition(), m_venue3.GetDefinition(), m_mic3.GetDefinition(), true);
*/

/*
	// Do I need this duplication if the ivid is already set in run request? 
	InstrumentVenueIdentification ivid(m_instrumentIdentifier.GetIdentifier(), m_venue.GetVenue(), m_mic.GetMIC());
	m_instrumentTradingInformation = tbricks::InstrumentTradingInformation(ivid);	
	tbricks::TickRule tickRule(m_instrumentTradingInformation.GetTickRule());
	tbricks::Volume roundLot(m_instrumentTradingInformation.GetLotSize());
	tbricks::Currency currency(m_instrumentTradingInformation.GetCurrency());
*/
	context.SendReply();
	TBDEBUG("<Predict::HandleValidateRequest()"); // << context);
}

/* 
 * KTK TODO - need to extend hedging logic to hedge on mulitple markets
 */
void 
Predict::HedgeWithLimits()
{

	if (m_positionStats.isFlat()) {
		m_legTimer.Stop();	
		return;
	}
	else {
		TBDEBUG("Not flat on timer - closing with limit");
		tbricks::Side hedgeSide;
		tbricks::Volume hedgeVolume;
		tbricks::Price lastPrice;
		tbricks::Price hedgePrice;
		m_positionStats.getHedgeSideAndVolume(hedgeSide, hedgeVolume);
		m_positionStats.getLastTradePrice(lastPrice);
		TBDEBUG("*** POSITION STATS *** - " << m_positionStats.ToString());	
		tbricks::Order::Options options;
		KAggregatedOrderBook::AggregateBBO bbo = m_pBBOBook->GetBBO();
		if (hedgeSide == tbricks::Side::BUY) {
			tbricks::Price hedgePrice = lastPrice;
			TBDEBUG("Placing limit buy: " << hedgeVolume << " @ "  << lastPrice << " PLUS TWO TICKS" );
			tbricks::InstrumentTradingInformation iti = m_ividToLeg[bbo.first.ivid]->GetInstrumentTradingInformation();
			iti.GetTickRule().Tick(hedgePrice, -2);
			if (!(hedgeVolume.Empty() || hedgePrice.Empty())) {
				m_ividToLeg[bbo.first.ivid]->SendOrder(hedgeVolume, hedgePrice , tbricks::Side::BUY, options);
			}
		}
		else {
			TBDEBUG("Placing limit sell: " << hedgeVolume << " @ "  << lastPrice << " PLUS TWO TICKS" );
			tbricks::InstrumentTradingInformation iti = m_ividToLeg[bbo.second.ivid]->GetInstrumentTradingInformation();
			iti.GetTickRule().Tick(hedgePrice, +2);
			if (!(hedgeVolume.Empty() || hedgePrice.Empty())) {
				m_ividToLeg[bbo.second.ivid]->SendOrder(hedgeVolume, hedgePrice , tbricks::Side::SELL, options);
			}
		}
		
	}
}

void
Predict::HandleTimerEvent(const Timer& timer)
{
	TBDEBUG(">Predict::HandleTimerEvent()");
	HedgeWithLimits();
	TBDEBUG("<Predict::HandleTimerEvent()");
}

/******************************************************************************
 * CALLBACKS FROM LEGS OF TRADES
 *****************************************************************************/

void 
Predict::Leg_HandleRecoveryCompleted(KSingleIVIDLeg& sender)
{
		
	TBDEBUG(">Predict::Leg_HandleTrade()");
	TBDEBUG("Recovered orders for leg: " << sender.GetIVID());
	TBDEBUG("<Predict::Leg_HandleTrade()");
}

/* Provides notification that a leg has traded - from interface ILegHandler
 */
void 
Predict::Leg_HandleTrade(KSingleIVIDLeg& sender, const Side& side, const Volume& tradeVolume, const Price& tradePrice, const tbricks::MIC& mic)
{
	TBDEBUG(">Predict::Leg_HandleTrade()");
	TBDEBUG("*** TRADE *** - SIDE: " << side << "; VOLUME: " << tradeVolume << "; PRICE: " << tradePrice);
	m_positionStats.Update(side, tradeVolume, tradePrice, mic, sender.GetIVID());
	if (!m_positionStats.isFlat())  {
		TBDEBUG("*** POSITION STATS *** - " << m_positionStats.ToString());	
		tbricks::Volume hedgeVolume;
		tbricks::Side hedgeSide; 
		m_positionStats.getHedgeSideAndVolume(hedgeSide, hedgeVolume);
		if (!m_sharedOrderManager.InRecovery()) {
			m_legTimer.Start(TimeInterval::Milliseconds(100));
		}
	}
	TBDEBUG("<Predict::Leg_HandleTrade()");
}

/******************************************************************************
 * CALLBACKS FROM LOCAL STREAMS
 *****************************************************************************/

void 
Predict::HandleStreamOpen(const tbricks::StreamIdentifier& stream)
{
	TBDEBUG("-Predict::HandleStreamOpen()");
	StreamStatus s; 
	
	m_streamState[stream] = s;
	m_streamState[stream].SetOpen();
}

void
Predict::HandleStreamClose(const tbricks::StreamIdentifier& stream)
{
	TBDEBUG("-Predict::HandleStreamClose()");
	tbricks::Hash<tbricks::StreamIdentifier, StreamStatus>::iterator it = m_streamState.find(stream);
	if (it != m_streamState.end()) {
		// Could SetClosed() on StreamStatus in hash here - but why? We're deleting.
		m_streamState.erase(stream);
	}
}

void 
Predict::HandleSnapshotEnd(const tbricks::StreamIdentifier& stream)
{
	TBDEBUG("-Predict::HandleSnapshotEnd()");
	tbricks::Hash<tbricks::StreamIdentifier, StreamStatus>::iterator it = m_streamState.find(stream);	
	if (it != m_streamState.end()) {
		m_streamState[stream].SetSnapshotDone();	
	}
}

void
Predict::HandleStreamStale(const tbricks::StreamIdentifier& stream)
{
	TBDEBUG("-Predict::HandleStreamStale()");
	tbricks::Hash<tbricks::StreamIdentifier, StreamStatus>::iterator it = m_streamState.find(stream);	
	if (it != m_streamState.end()) {
		m_streamState[stream].SetStale();	
	}
}

/* Provides notification of position change - used only for reconciliation - that is not for real-time position
 * keeping since real-time position should be computed by trades themselves. Handle position requires a round
 * trip to the positions database of which there is only one - in a distributed system this may cause delays
 * depending on where the position server is located so it's best to use this just to see if our position matches
 * the "delayed" position in the position server db 
 */
void 
Predict::HandlePosition(const StreamIdentifier& stream, const Position& position)
{
	TBDEBUG("-Predict::HandlePosition()");
	TBDEBUG("Position: " << position);
	//tbricks::Currency cur = position.GetCurrency();
	//tbricks::Double valueBought = position.GetGrossTradeAmountBought();
	//tbricks::Double valueSold = position.GetGrossTradeAmountSold();
	//tbricks::Double volumeBought = position.GetVolumeBought();
	//tbricks::Double volumeSold = position.GetVolumeSold();
	//tbricks::Double netPos = volumeBought - volumeSold;
	//tbricks::Double cashPos = valueSold - valueBought;
}

/* Called to indicate that a position that previously matched the stream filter no longer matches - e.g. 
 * if stream was opened with filter for all active orders and an order that previously matched the filter
 * becomes filled then invalidate is called
 */
void 
Predict::HandlePositionInvalidate(const StreamIdentifier& stream, const Identifier& id)
{
	TBDEBUG("-Predict::HandlePositionInvalidate()");
}

/******************************************************************************
 * CALLBACKS FROM AGGREGATED ORDER BOOK
 *****************************************************************************/

/* This will receive a callback from KAggregatedOrderBook when the AGGREGATED BBO changes - not any individual BBO.
 * IF the individual BBO does't beat the current aggregated best BBO then this will not be called
 */
void 
Predict::KAggregatedOrderBook_HandleBestPrice(const KAggregatedOrderBook::PriceAndIVID& bestBid, const KAggregatedOrderBook::PriceAndIVID& bestAsk)
{
    
    static std::vector<float> bids_vec; 
    static std::vector<uint32_t> bid_sizes_vec; 
    static std::vector<float> offers_vec; 
    static std::vector<uint32_t> offer_sizes_vec;  
    
	TBDEBUG(">Predict::KAggregatedOrderBook_HandleBestPrice()"); 
	// Should do the following instead
	// Get tick rule
	// Use that to compare prices
	if (!(bestBid.price.Empty() && bestAsk.price.Empty())) {
		if ((bestBid.price > bestAsk.price) && (bestBid.mic != bestAsk.mic)) {
			TBDEBUG("CROSSED MARKET: " << bestBid.mic << "=" << bestBid.volume << "@" << bestBid.price << "||" << bestAsk.mic << "=" << bestAsk.volume  << "@" << bestAsk.price);
			if (!m_pBBOBook->IsConsistent()) {
				TBWARNING("Aggregated orderbook is not consistent - NOT sending orders");
				return;
			}
			if (bestBid.price.Empty() || bestBid.volume.Empty() || bestAsk.price.Empty() || bestAsk.volume.Empty()) {
				TBWARNING("A price or volume is empty - NOT sending orders");
				return;
			}
            
            // if we only have best bid and best offer then 
            // pass 1-element arrays to python predictor 
            bids_vec.clear();
            bids_vec[0] = bestBid.price; 
            
            bid_sizes_vec.clear(); 
            bid_sizes_vec[0] = bestBid.volume; 
            
            offers_vec.clear(); 
            offers_vec[0] = bestAsk.price; 
            
            offer_sizes_vec.clear(); 
            offer_sizes_vec[0] = bestAsk.volume; 
            static unsigned int t = 0; //TODO: get the time 
            t += 1; 
            this->predictor->update(t, bids_vec, bid_sizes_vec, offers_vec, offer_sizes_vec); 
            
            
			// Don't try another one until this current one is flat
			if (m_positionStats.isFlat()) {
				tbricks::Price margin;
				margin = bestBid.price - bestAsk.price;
				// Check to see how much the cross is first - too small or too big and don't take the shot
				if ((margin > 2*ONE_PIP) && (margin < 5*ONE_PIP)) {
					tbricks::Order::Options options;
					options.SetValidity(tbricks::Validity(tbricks::Validity::VALID_IMMEDIATE));
					// KTK TODO - adjust volume to a more rational figure
					m_ividToLeg[bestBid.ivid]->SendOrder(tbricks::Volume(500000), bestBid.price, tbricks::Side::SELL, options);
					m_ividToLeg[bestAsk.ivid]->SendOrder(tbricks::Volume(500000), bestAsk.price, tbricks::Side::BUY, options);
				}
				else {
					TBWARNING("Margin outside threshold - NOT sending orders");
				}
			}	
			else {
				TBWARNING("Position not flat - NOT sending orders");
			}
		}	
	}

	TBDEBUG("<Predict::KAggregatedOrderBook_HandleBestPrice()");
}

void 
Predict::KAggregatedOrderBook_HandleInstrumentStatus(const tbricks::InstrumentStatus& status) 
{
	TBDEBUG("-Predict::KAggregatedOrderBook_HandleInstrumentStatus()");
}

void 
Predict::KAggregatedOrderBook_HandleStatistics(const tbricks::Statistics& stats) 
{
	TBDEBUG("-Predict::KAggregatedOrderBook_HandleStatistics()");
}

void 
Predict::KAggregatedOrderBook_HandleStreamStale(const StreamIdentifier& stream) 
{
	TBDEBUG("-Predict::KAggregatedOrderBook_HandleStreamStale()");
/*
	if (m_allSnapshotsComplete) {
		if (m_bboStream1 == stream || m_bboStream2 == stream || m_bboStream3 == stream) {
			TBDEBUG("====> A best price stream is stale - IGNORING");
			if (stream == m_bboStream3.GetIdentifier()) {
				TBSTATUS("====> Likely FX Stream shows stale prices- IGNORING");
			}
		}
		else if (m_statsStream1 == stream || m_statsStream2 == stream || m_statsStream3 == stream) {
			TBDEBUG("====> A statistics stram is stale - PAUSING");
			if (stream == m_statsStream3.GetIdentifier()) {
				TBSTATUS("====> Likely FX Stream shows stale stats - IGNORING");
			}
			//HandlePauseRequest(); 
		}
		else if (m_instStatusStream1 == stream || m_instStatusStream2 == stream || m_instStatusStream3 == stream) {
			TBDEBUG("====> An instrument staus stream is stale - IGNORING");
		}
	}
*/
}

void 
Predict::KAggregatedOrderBook_HandleStreamClose(const StreamIdentifier & stream)
{
	TBDEBUG("-Predict::KAggregatedOrderBook_HandleStreamClose()");
}

void 
Predict::KAggregatedOrderBook_HandleStreamOpen(const StreamIdentifier & stream) 
{
	TBDEBUG("-Predict::KAggregatedOrderBook_HandleStreamOpen()");
}


// KTK - FIX this simply counts the number of open streams - doesn't tell when snapshot for 
// all streams are complete
void 
Predict::KAggregatedOrderBook_HandleSnapshotEnd(const StreamIdentifier & stream)
{
	TBDEBUG("-Predict::KAggregatedOrderBook_HandleSnapshotEnd()");

}

void 
Predict::PrintState()
{
	//TBDEBUG("Pending state: <" << m_pendingState << "> current state: <" << GetState() << "> pending requests: <" << m_pendingRequests.size() << "> order count: <" << m_rawOrders.size() << ">");
	TBDEBUG("-Predict::PrintState() - Current state: " << GetState()); 

}

/*
void Predict::HandleRequestReply(const Identifier & request_id, Status status, const String & text)
{
	TBDEBUG("HandleRequestReply()");
	if (status == tbricks::Status::FAIL) {
		TBSTATUS("!!!!> Request failed: " << request_id << "text is <" << text << ">");
		SetTransactionFail("!!!!> HandleRequestReply - Failed to set state - order status text = <" + text + ">", StrategyState::PAUSED);
		//m_pendingRequests.erase(request_id);	
		//HandlePauseRequest();
	}
	if (status == tbricks::Status::SERVICE_WENT_DOWN) {
		TBSTATUS("!!!!> Request failed - service went down: " << request_id << "text is <" << text << ">");
		SetTransactionFail("!!!!> HandleRequestReply - Failed to set state - order status text = <" + text + ">", StrategyState::PAUSED);
		//HandlePauseRequest();
	}
	if (status == tbricks::Status::OK) {
		PendingRequestHashIterator it = m_pendingRequests.find(request_id);
		if (it != m_pendingRequests.end()) {
			m_pendingRequests.erase(request_id);
		}	
		else {
			TBSTATUS("!!!!> No pending request for id:  - request_id: " << request_id);
		}
	}
	// KTK - removed 8/27/2010 - don't think it's needed to update state here since only transaction failure will impact state and that will get set when transaction fails as per second parameter to SetTransactionFail(...);
	//UpdateState();
	TBDEBUG("HandleRequestReply()");
}
*/

/******************************************************************************
 * STATE MANAGEMENT
 *****************************************************************************/

/*
void Predict::UpdateState(void)
{
	//PrintState();
	//TBDEBUG("UpdateState");	
	//KTK - ??????????????
	//if (m_pendingState == tbricks::StrategyState::NONE) {
		//SetState(StrategyState::RUNNING);
	//}
	
	if (m_pendingState == tbricks::StrategyState::RUNNING) {
		//if (InRecovery() == true) {
			//TBDEBUG("Still in recovery - not setting RUNNING yet");
			//return;
		//}
		//else {
			SetState(StrategyState::RUNNING);
			m_pendingState = StrategyState::NONE;
		//}
	}

	if (m_pendingState == tbricks::StrategyState::DELETED) {
		SetState(StrategyState::DELETED);
		m_pendingState = StrategyState::NONE;
	}
	if (m_pendingState == tbricks::StrategyState::PAUSED) {
		SetState(StrategyState::PAUSED);
		m_pendingState = StrategyState::NONE;
	}

	TBDEBUG("UpdateState");	
}
*/

void Predict::UpdateParameters(const StrategyParameters & parameters)
{	
	TBDEBUG("UpdateParameters");
	m_instrumentParam.GetParameter(parameters);
	
	m_portfolio.GetParameter(parameters);
       	
	// Don't really need a vec of instruments but it's not time critical
	m_instruments.push_back(m_instrumentParam.GetInstrumentIdentifier());

	// KTK - set the instruments associated with the strategy
	SetInstruments(m_instruments);
	TBDEBUG("UpdateParameters");
}

 

static void GetParameterDefinitions(std::vector<tbricks::ParameterDefinition> & definitions)
{
    definitions.push_back(strategy_parameters::Instrument());
    definitions.push_back(strategy_parameters::MaxVolume());
    definitions.push_back(strategy_parameters::Portfolio());
}


DEFINE_STRATEGY_ENTRY(Predict)
