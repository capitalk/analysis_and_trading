/*
 *  main.h
 *
 * 
 */

#ifndef __KTK_STRATEGY_eb8db328_38e1_11df_92ac_07732616a5a7__H
#define __KTK_STRATEGY_eb8db328_38e1_11df_92ac_07732616a5a7__H

// TBricks
#include "strategy/API.h"

#include "InstrumentPosition.h"

// CapK
#include "KAggregatedOrderBook.h"
#include "KSingleIVIDLeg.h"
#include "Utils.h"
#include "KPositionStatistics.h"

// STL/Boost
#include <vector>

#include "predictor.h" 

class Predict:  public tbricks::Strategy,
//public tbricks::BestPriceStream::IHandler,
//public tbricks::StatisticsStream::IHandler,
//public tbricks::InstrumentStatusStream::IHandler,
//Module::IModuleHandler,
tbricks::ITimerEventHandler,
tbricks::PositionStream::IHandler,
KSingleIVIDLeg::ILegListener,
KAggregatedOrderBook::IKAggregatedOrderBookHandler
{
public:
    Predict(const tbricks::InitializationReason & reason, const tbricks::StrategyParameters & parameters);
    ~Predict();
		
	// Strategy interfaces 
	virtual void HandleDeleteRequest(void);
	virtual void HandleRunRequest(void);
	virtual void HandlePauseRequest(void);
	virtual void HandleModifyRequest(const tbricks::StrategyModifier & modifier);
	virtual void HandleValidateRequest(tbricks::ValidationContext &context);

	// Aggregated OrderBook interfaces
	virtual void KAggregatedOrderBook_HandleStreamOpen(const tbricks::StreamIdentifier & stream);
	virtual void KAggregatedOrderBook_HandleStreamClose(const tbricks::StreamIdentifier & stream);
	virtual void KAggregatedOrderBook_HandleSnapshotEnd(const tbricks::StreamIdentifier & stream);
	virtual void KAggregatedOrderBook_HandleStreamStale(const tbricks::StreamIdentifier & stream);
	virtual void KAggregatedOrderBook_HandleStatistics(const tbricks::Statistics & statistics);
	virtual void KAggregatedOrderBook_HandleInstrumentStatus(const tbricks::InstrumentStatus & status);
	virtual void KAggregatedOrderBook_HandleBestPrice(const KAggregatedOrderBook::PriceAndIVID& bestBid, const KAggregatedOrderBook::PriceAndIVID& bestAsk);

	// PositionStream interfaces	
	virtual void HandlePosition(const StreamIdentifier& stream, const Position& position);
	virtual void HandlePositionInvalidate(const StreamIdentifier& stream, const Identifier & id);

	// Stream interfaces as applied to streams opened in this class - i.e. not necessarily KAggregatedOrderBook since  
	// there are different callbacks for those streams
	virtual void HandleStreamOpen(const tbricks::StreamIdentifier & stream);
	virtual void HandleStreamClose(const tbricks::StreamIdentifier & stream);
	virtual void HandleSnapshotEnd(const tbricks::StreamIdentifier & stream);
	virtual void HandleStreamStale(const tbricks::StreamIdentifier & stream);

	// ILegListener
	//virtual void Leg_Trade(KLeg::IKLegListener& sender, const Side& side, const Volume& tradeVolume, const Price& tradePrice);
	virtual void Leg_HandleTrade(KSingleIVIDLeg& sender, const Side& side, const Volume& tradeVolume, const Price& tradePrice, const tbricks::MIC& mic);
	virtual void Leg_HandleRecoveryCompleted(KSingleIVIDLeg& sender);

	// ITimerEventHandler
	void HandleTimerEvent(const tbricks::Timer& timer);

private:
    PythonPredictor* predictor; 
    
	void HedgeWithLimits();

	
	// Parameters
	tbricks::InitializationReason m_initializationReason;
	tbricks::InstrumentIdentifierParameter m_instrumentParam;
	tbricks::PortfolioIdentifierParameter m_portfolio;

	// Instrument information
	tbricks::Instrument m_instrument;
	tbricks::InstrumentIdentifier m_instrumentIdentifier;
	tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::InstrumentTradingInformation> m_ITI;

	// Orderbook
	KAggregatedOrderBook* m_pBBOBook;

	// Streams
	tbricks::PositionStream m_positionStream;

	// Stream management
	//std::vector<tbricks::StreamIdentifier > m_openStreams; 
	tbricks::Hash<tbricks::StreamIdentifier, StreamStatus> m_streamState;
	std::vector<tbricks::InstrumentIdentifier> m_instruments;

	tbricks::Double m_valueFactor;

	// Order Management
	SharedOrderManager m_sharedOrderManager;
	typedef std::vector<tbricks::InstrumentVenueIdentification> IVIDVector;
	IVIDVector m_tradingIvids;
	IVIDVector m_marketIvids;
	tbricks::Hash<tbricks::InstrumentVenueIdentification, KSingleIVIDLeg*> m_ividToLeg;
	void DeleteAllOrders(void);
	void DestroyLegs(void);
	tbricks::Timer m_legTimer;

	// State management
	bool m_allSnapshotsComplete; 
	bool m_isActive;
	tbricks::StrategyState m_pendingState;
	void PrintState();
	
	// Position information
	KPositionStatistics m_positionStats;
	

	void UpdateParameters(const tbricks::StrategyParameters & parameters);
	
	//void UpdateState(void);
	void CloseAllStreams(void);

};

#endif /* __KTK_STRATEGY_eb8db328_38e1_11df_92ac_07732616a5a7__H */

//****************************************************************************
