 
#include "KAggregatedOrderBook.h"

KAggregatedOrderBook::KAggregatedOrderBook(const tbricks::Instrument &inst):_inst(inst), _streamOpenRequests(0), _streamOpenCallbacks(0),  _streamSnapshotCompleteCallbacks(0)
{
	if (inst.Empty()) {
		TBWARNING("KAggregatedOrderBook: Instrument is empty");
	}
	if (false == _inst.Resolve(_inst.GetIdentifier())) {
		TBWARNING("KAggregatedOrderBook: Resolve failed" << __FILE__ << __LINE__);
	}
}

KAggregatedOrderBook::~KAggregatedOrderBook(void)
{
	for (IvidToInstStatus::iterator i = _ividToInstStatus.begin(); i != _ividToInstStatus.end(); i++) {
		delete (i->second);
	}
	for (IvidToStatistics::iterator i = _ividToStatistics.begin(); i != _ividToStatistics.end(); i++) {
		delete (i->second);
	}
	for (IvidToBestPrice::iterator i = _ividToBestPrice.begin(); i != _ividToBestPrice.end(); i++) {
		delete (i->second);
	}
}
		
void 
KAggregatedOrderBook::AddHandler(IKAggregatedOrderBookHandler* handler) 
{
	TBDEBUG("KAggregatedOrderBook::AddHandler()");
	_handlers.insert(handler);
}

void 
KAggregatedOrderBook::RemoveHandler(IKAggregatedOrderBookHandler* handler) 
{
	TBDEBUG("KAggregatedOrderBook::RemoveHandler()");
	Handlers::iterator i = _handlers.find(handler);
	if (i != _handlers.end()) {
		_handlers.erase(i);
	}
}

void
KAggregatedOrderBook::Open(tbricks::Stream::Type t, bool suppressVolumeUpdates) // const tbricks::StreamOptions& options)
{
	if (IsOpen()) {
		return;
	}
	TBDEBUG("KAggregatedOrderBook::Open()");
	if (_inst.Empty()) {
		TBWARNING("Cant' open instrument - it is empty");
		return;
	}

	_inst.GetTradingIdentifications(_tradingIvids);
	_inst.GetMarketDataIdentifications(_marketIvids);
	TBDEBUG("Found " << _tradingIvids.size() << " for trading IVID");
	TBDEBUG("Found " << _marketIvids.size() << " for market IVID");
	for (IVIDVector::iterator i = _tradingIvids.begin(); i != _tradingIvids.end(); i++) {
		_ividToInstStatus[(*i)] = new tbricks::InstrumentStatusStream();	
		_ividToInstStatus[(*i)]->Open(tbricks::Stream::SNAPSHOT_AND_LIVE, (*i));	
		_pendingSnapshots.insert(_ividToInstStatus[(*i)]->GetIdentifier());
		_streamOpenRequests++;
			
		_ividToStatistics[(*i)] = new tbricks::StatisticsStream(); 	
		_ividToStatistics[(*i)]->Open(tbricks::Stream::SNAPSHOT_AND_LIVE, (*i));	
		_pendingSnapshots.insert(_ividToStatistics[(*i)]->GetIdentifier());
		_streamOpenRequests++;

		_ividToBestPrice[(*i)] = new tbricks::BestPriceStream();	
		_ividToBestPrice[(*i)]->SetPriority(100);
		_ividToBestPrice[(*i)]->Open(tbricks::Stream::SNAPSHOT_AND_LIVE, (*i), AnyBestPriceFilter(), false /* coalescing*/);	
		_pendingSnapshots.insert(_ividToStatistics[(*i)]->GetIdentifier());
		_streamOpenRequests++;
		

		// Set the stream identifier and an empty best price
		// for use later in handle best price - the map of stream to BBO will cache price info
		// and maintain ivid 
		StreamIdentifier streamId = _ividToBestPrice[(*i)]->GetIdentifier();
		BestPriceAndIVID bpivid = BestPriceAndIVID();
		bpivid.bestPrice = tbricks::BestPrice();
		bpivid.ivid = *i;
		_streamToBBO[streamId] = bpivid;
	}
	TBDEBUG("Requested streams open: " << _streamOpenRequests);
}

void
KAggregatedOrderBook::Close()
{
	for (IvidToInstStatus::iterator i = _ividToInstStatus.begin(); i != _ividToInstStatus.end(); i++) {
		i->second->Close();
	}
	for (IvidToStatistics::iterator i = _ividToStatistics.begin(); i != _ividToStatistics.end(); i++) {
		i->second->Close();
	}
	for (IvidToBestPrice::iterator i = _ividToBestPrice.begin(); i != _ividToBestPrice.end(); i++) {
		i->second->Close();
	}
}

bool
KAggregatedOrderBook::IsOpen(void) 
{
	for (IvidToInstStatus::iterator i = _ividToInstStatus.begin(); i != _ividToInstStatus.end(); i++) {
		if (i->second->IsOpen()) {
			return true;
		}
	}
	for (IvidToStatistics::iterator i = _ividToStatistics.begin(); i != _ividToStatistics.end(); i++) {
		if (i->second->IsOpen()) {
			return true;
		}
	}
	for (IvidToBestPrice::iterator i = _ividToBestPrice.begin(); i != _ividToBestPrice.end(); i++) {
		if (i->second->IsOpen()) {
			return true;
		}
	}
	return false;
} 

void 
KAggregatedOrderBook::HandleStreamOpen(const tbricks::StreamIdentifier& stream)
{
	Handlers::iterator it;
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleStreamOpen(stream);
	}
	_streamOpenCallbacks++;

}

void 
KAggregatedOrderBook::HandleStreamClose(const tbricks::StreamIdentifier& stream)
{
	TBWARNING("Stream closed: " << stream.GetUuid());
	Handlers::iterator it;
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleStreamClose(stream);
	}
}

void 
KAggregatedOrderBook::HandleSnapshotEnd(const tbricks::StreamIdentifier& stream)
{
	TBDEBUG("Snapshot end: " << stream.GetUuid());
	Handlers::iterator it;
	_pendingSnapshots.erase(stream);
	TBDEBUG(_pendingSnapshots.size() << " snapshots pennding after this snapshot");
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleSnapshotEnd(stream);
	}
	_streamSnapshotCompleteCallbacks++;
}

void 
KAggregatedOrderBook::HandleStreamStale(const tbricks::StreamIdentifier& stream)
{
	TBWARNING("Stream stale: " << stream.GetUuid());
	Handlers::iterator it;
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleStreamStale(stream);
	}
}

void
KAggregatedOrderBook::HandleBestPrice(const tbricks::StreamIdentifier& stream, const tbricks::BestPrice& update)
{

	// KTK - it is safer to do this with find but faster without and we 
	// need to assume that the stream is open and exists - so we'll just access it directly
	// with operator[] for now. 
	//StreamToBBO::iterator i = _streamToBBO.find(stream);
	//if (i != _streamToBBO.end()) {
	
		if (update.BidPriceUpdated()) { TBDEBUG("Bid price updated"); }
		if (update.BidVolumeUpdated()) { TBDEBUG("Bid volume updated"); }
		if (update.AskPriceUpdated()) { TBDEBUG("Ask price updated"); }
		if (update.AskVolumeUpdated()) { TBDEBUG("Ask volume updated"); }
		_streamToBBO[stream].bestPrice.Merge(update);
	//}
	//else {
		//TBWARNING("No matching stream found for best price! Stream ID: " << stream);
	//}
	StreamToBBO::iterator i;
	for (i = _streamToBBO.begin(); i != _streamToBBO.end(); i++) {
		if (i->second.bestPrice.GetBidPrice() > _bestBid.price) {
		//if (_streamToBBO[stream].bestPrice.GetBidPrice() > _bestBid.price) {
			_bestBid.price = _streamToBBO[stream].bestPrice.GetBidPrice();
			_bestBid.volume = _streamToBBO[stream].bestPrice.GetBidVolume();
			_bestBid.mic = _streamToBBO[stream].ivid.GetMIC();
			_bestBid.ivid = _streamToBBO[stream].ivid;
		}	
		if (i->second.bestPrice.GetAskPrice() > _bestAsk.price) {
		//if (i->second.bestPrice.GetAskPrice() < _bestAsk.price) {
			_bestAsk.price = _streamToBBO[stream].bestPrice.GetAskPrice();
			_bestAsk.volume = _streamToBBO[stream].bestPrice.GetAskVolume();
			_bestAsk.mic = _streamToBBO[stream].ivid.GetMIC();
			_bestAsk.ivid = _streamToBBO[stream].ivid;
		}
	}
	Handlers::iterator it;
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleBestPrice(_bestBid, _bestAsk);
	}


	// Maybe a faster option - don't cache prices at all...
	// but then how do we deal with updates that have only price or volume 
	// and where do we get MIC from?
	/* 
	if (bestBid.Empty() && update.BidPriceUpdated()) {
		bestBid.price = update.GetBidPrice();
	}
	else if (!update.GetBidPrice().Empty()) {
		if (update.GetBidPrice() > bestBid.price) {
			bestBid.price = update.GetBidPrice();
		}
	}	
	*/
}

void
KAggregatedOrderBook::HandleInstrumentStatus(const tbricks::StreamIdentifier& stream, const tbricks::InstrumentStatus& status)
{
	Handlers::iterator it;
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleInstrumentStatus(status);
	}
}

void
KAggregatedOrderBook::HandleStatistics(const tbricks::StreamIdentifier& stream, const tbricks::Statistics& statistics)
{
	Handlers::iterator it;
	for (it = _handlers.begin(); it != _handlers.end(); it++) {
		(*it)->KAggregatedOrderBook_HandleStatistics(statistics);
	}
}

void 
KAggregatedOrderBook::getInstrumentTradingInformation(tbricks::Hash<tbricks::InstrumentVenueIdentification, tbricks::InstrumentTradingInformation>& ITIs)
{
	ITIs = _ividToITI;
}
