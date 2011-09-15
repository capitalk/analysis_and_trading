
#include "strategy/API.h"
#include "strategy/stream/StreamOptions.h"
#include "shared/Helpers.h"

#ifndef __AGGREGATED_ORDERBOOK_H__
#define __AGGREGATED_ORDERBOOK_H__

class KAggregatedOrderBook:
	tbricks::InstrumentStatusStream::IHandler,
	tbricks::StatisticsStream::IHandler,
	tbricks::BestPriceStream::IHandler
{
	public: 
		
		// Individual best bid and ask across all MICs
		struct s_PriceAndIVID
		{
			tbricks::Price price;
			tbricks::Volume volume;
			tbricks::MIC mic;
			tbricks::InstrumentVenueIdentification ivid;
		};
		typedef struct s_PriceAndIVID PriceAndIVID;

		// Return type for getting aggregated BBO across multiple books
		typedef std::pair<PriceAndIVID, PriceAndIVID> AggregateBBO;

		// Cache the ivid info as well as best price when updated from BBO stream
		struct s_BestPriceAndIVID
		{
			tbricks::BestPrice bestPrice;
			tbricks::InstrumentVenueIdentification ivid;
		};
		typedef struct s_BestPriceAndIVID BestPriceAndIVID;

		class IKAggregatedOrderBookHandler
		{
			public:
				virtual void KAggregatedOrderBook_HandleStreamOpen(const tbricks::StreamIdentifier& stream) {};
				virtual void KAggregatedOrderBook_HandleStreamClose(const tbricks::StreamIdentifier& stream) {};
				virtual void KAggregatedOrderBook_HandleSnapshotEnd(const tbricks::StreamIdentifier& stream) {};
				virtual void KAggregatedOrderBook_HandleStreamStale(const tbricks::StreamIdentifier& stream) {};
				virtual void KAggregatedOrderBook_HandleStatistics(const tbricks::Statistics& statistics) {};
				virtual void KAggregatedOrderBook_HandleInstrumentStatus(const tbricks::InstrumentStatus& status) = 0;
				virtual void KAggregatedOrderBook_HandleBestPrice(const PriceAndIVID& bestBid, const PriceAndIVID& bestAsk) = 0;
		};

		// ctor and dtor
		KAggregatedOrderBook(const tbricks::Instrument  &inst);
		virtual ~KAggregatedOrderBook(void);
		
		// Add/remove the objects that want to receive notifications about actions on any of the orderbooks that 
		// are opened for the KAggregatedOrderBook
		void AddHandler(IKAggregatedOrderBookHandler* handler);
		void RemoveHandler(IKAggregatedOrderBookHandler* handler);

		// Open(SnapshotChoice - i.e. SNAPSHOT_AND_LIVE or just SNAPSHOT, Options - i.e. filters and throttles)
		void Open(tbricks::Stream::Type t, bool suppressVolumeUpdates = true);// tbricks::StreamOptions& options);

		// Closes all streams
		void Close(void);

		// Will return true iff at least one stream is open
		bool IsOpen(void);

		// Return the best bid and ask prices from the aggregated book
		AggregateBBO GetBBO() { return std::make_pair(_bestBid, _bestAsk); }

		// Return true if the number of streams opened == number of callbacks received == snapshot end callbacks
		inline bool IsConsistent(void) { return ((_streamOpenRequests == _streamOpenCallbacks) && (_streamOpenCallbacks == _streamSnapshotCompleteCallbacks));}

		// Local stream callbacks
		virtual void HandleStreamOpen(const tbricks::StreamIdentifier& stream);
		virtual void HandleStreamClose(const tbricks::StreamIdentifier& stream);
		virtual void HandleSnapshotEnd(const tbricks::StreamIdentifier& stream);
		virtual void HandleStreamStale(const tbricks::StreamIdentifier& stream);
		virtual void HandleStatistics(const tbricks::StreamIdentifier& stream, const tbricks::Statistics& statistics);
		virtual void HandleInstrumentStatus(const tbricks::StreamIdentifier& stream, const tbricks::InstrumentStatus& status);
		virtual void HandleBestPrice(const tbricks::StreamIdentifier& stream, const tbricks::BestPrice& update);

		//  Get which instrument streams we're listening to
		std::vector<tbricks::InstrumentVenueIdentification>  getTradingIvids() { return _tradingIvids;};
		std::vector<tbricks::InstrumentVenueIdentification>  getMarketDataIvids() { return _marketIvids;};
		void getInstrumentTradingInformation(tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::InstrumentTradingInformation>& ITIs);

	protected:
		tbricks::Instrument _inst;
/*
		tbricks::Hash<const tbricks::StreamIdentifier, tbricks::Statistics> _streamToStatistics;
		tbricks::Hash<const tbricks::StreamIdentifier, tbricks::BestPrice> _streamToBestPrice;
		tbricks::Hash<const tbricks::StreamIdentifier, const tbricks::InstrumentIdentifier> _streamToIvid;
		tbricks::Hash<const tbricks::StreamIdentifier, const tbricks::InstrumentVenueIdentification> _streamToIvid;
*/
/*
		struct s_StreamSet 
		{
			tbricks::InstrumentStatusStream instStatusStream;
			tbricks::StatisticsStream statisticsStream;
			tbricks::BestPriceStream bestPriceStream;
		};
		typedef struct s_StreamSet StreamSet;
		typedef tbricks::Hash<const tbricks::InstrumentVenueIdentification, StreamSet> Streams;
		Streams _ividToStream;
*/
		

		// Hold the best price and MIC AMONG ALL STREAMS for BID and ASK separately
		PriceAndIVID _bestBid;
		PriceAndIVID _bestAsk;

		typedef tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::InstrumentStatusStream*> IvidToInstStatus;
		typedef tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::StatisticsStream*> IvidToStatistics;
		typedef tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::BestPriceStream*> IvidToBestPrice;
		typedef tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::InstrumentTradingInformation> IvidToITI;
		typedef tbricks::Hash<tbricks::StreamIdentifier, BestPriceAndIVID> StreamToBBO;
		IvidToInstStatus _ividToInstStatus;
		IvidToStatistics _ividToStatistics;
		IvidToBestPrice _ividToBestPrice;
		IvidToITI _ividToITI;
		StreamToBBO _streamToBBO;

	private:	
		typedef std::set<IKAggregatedOrderBookHandler*> Handlers;
		Handlers _handlers;
		size_t _streamOpenRequests;
		size_t _streamOpenCallbacks;
		size_t _streamSnapshotCompleteCallbacks;
		
		typedef std::vector<tbricks::InstrumentVenueIdentification> IVIDVector;
		IVIDVector _marketIvids;
		IVIDVector _tradingIvids;

		typedef std::set<tbricks::StreamIdentifier> StreamIDSet;
		StreamIDSet _pendingSnapshots;
	
};

#endif // __AGGREGATED_ORDERBOOK_H__
