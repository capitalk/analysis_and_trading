
#include "KSingleIVIDLeg.h"

KSingleIVIDLeg::KSingleIVIDLeg(const tbricks::Strategy& parentStrategy, 
	KSingleIVIDLeg::ILegListener& listener,
	SharedOrderManager& orderManager,
	const tbricks::InstrumentVenueIdentification& ivid,
	const tbricks::PortfolioIdentifier& portfolio,
	const tbricks::String& name):
m_strategy(parentStrategy),
m_orderManager(orderManager),
m_listener(listener),
m_ivid(ivid),
m_portfolioId(portfolio),
m_name(name),
m_ITI(ivid),
m_longVol(0),
m_shortVol(0),
m_isActive(false)
{
	TBDEBUG(">KSingleIVIDLeg::KSingleIVIDLeg()");
	m_orderManager.AddManager(*this);
	m_instrument = tbricks::Instrument(m_ivid.GetInstrumentIdentifier());
	m_mic = m_ivid.GetMIC();
	TBDEBUG("<KSingleIVIDLeg::KSingleIVIDLeg()");
}

KSingleIVIDLeg::~KSingleIVIDLeg(void)
{
	TBDEBUG(">KSingleIVIDLeg::~KSingleIVIDLeg()");
	m_orderManager.RemoveManager(*this);
	TBDEBUG("<KSingleIVIDLeg::~KSingleIVIDLeg()");
}

void
KSingleIVIDLeg::SetActive(bool active)
{
	if (active == false) CancelAll(); m_isActive = active; 
}

void KSingleIVIDLeg::HandleOrderUpdate(const tbricks::Order::Update& update) 
{
	TBDEBUG(">KSingleIVIDLeg::HandleOrderUpdate()" << update);
	OrderIdToOrderInfo::iterator o = m_orders.find(update.GetIdentifier());
	// Bark if we can't find the order
	if (o == m_orders.end()) {
		TBSTATUS("----> Got update for unknown order:" << update);
		return;
	}
	// Clear peding volume for order since we should get it in the update now
	o->second.m_pendingVolume = 0;
	

	if (update.HasFilledVolume()) {
		tbricks::Volume newFilledVolume;
		update.GetFilledVolume(newFilledVolume);
		
		tbricks::Volume oldFilledVolume;
		if (o->second.m_order.HasFilledVolume()) {
			o->second.m_order.GetFilledVolume(oldFilledVolume);
		}	
		else {
			oldFilledVolume = 0;
		}

		// Filled volume should monotonically increase whether long OR short
		// Remember that we're ASSUMING the price here we got filled at 
		// since the trade conf may come after the fill confirm
		tbricks::Volume tradeVolume = newFilledVolume - oldFilledVolume;
		tbricks::Price tradePrice;
		if (m_ITI.CompareVolumes(tradeVolume.GetDouble(), 0) == tbricks::TB_GREATER_THAN) { 
			if (o->second.m_options.GetSide() == tbricks::Side::BUY) {
				m_longVol += tradeVolume;

				if (update.HasPrice()) { 
					update.GetPrice(tradePrice);
				}
				else {
					o->second.m_order.GetPrice(tradePrice);
				}
				TBSTATUS("----> TRADE (B): " << tradeVolume << m_instrument.GetShortName() << " @ " << tradePrice);
				// Inform listeners of trades
				m_listener.Leg_HandleTrade(*this, tbricks::Side::BUY, tradeVolume, tradePrice, m_mic);
			}	
			else {
				m_shortVol += tradeVolume;
				
				if (update.HasPrice()) {
					update.GetPrice(tradePrice);
				}
				else {
					o->second.m_order.GetPrice(tradePrice);
				}
				TBSTATUS("----> TRADE (S): " << tradeVolume << m_instrument.GetShortName() << " @ " << tradePrice);
				// Inform listeners of trades
				m_listener.Leg_HandleTrade(*this, tbricks::Side::BUY, tradeVolume, tradePrice, m_mic);
			}
		}
	}

	o->second.m_order.Merge(update);	

	// Check to see if the order is deleted 
	// if so remove from the set of orders	outstanding
	tbricks::Boolean isDeleted;	
	update.GetDeleted(isDeleted);
	if (update.HasDeleted() && isDeleted) {
		TBDEBUG("----> Deleted order " << update.GetIdentifier());
		m_orders.erase(o);
	}	

	
	if (update.HasTransactionState()) {
		tbricks::TransactionState ts;
		update.GetTransactionState(ts);

		if (ts == tbricks::TransactionState::FAIL) {
			tbricks::String failText;
			update.GetStatusText(failText);
		}
	}	
	TBDEBUG("<KSingleIVIDLeg::HandleOrderUpdate()" << update);
}

void KSingleIVIDLeg::HandleRecoveryCompleted() 
{
	TBDEBUG(">KSingleIVIDLeg::HandleRecoveryCompleted()");
	TBDEBUG("<KSingleIVIDLeg::HandleRecoveryCompleted()");
}

void KSingleIVIDLeg::HandleRequestReply(const tbricks::Identifier& requestId, tbricks::Status status, const tbricks::String& statusText) 
{
	TBDEBUG(">KSingleIVIDLeg::HandleRequestReply()");// id: " << requestId << " status: " << status);
	// KTK - first check if it's in the deleted list - then check for failure

	RequestToOrderId::iterator request = m_requestToOrder.find(requestId);
	// Request not found for order
	if (request == m_requestToOrder.end()) {
		TBDEBUG("Request NOT found for order - checking deletes");
		RequestToOrderId::iterator deleteRequest = m_deleteRequestToOrder.find(requestId);

		if (deleteRequest != m_deleteRequestToOrder.end()) {
			TBDEBUG("Request FOUND in deletes ");

			m_deleteRequestToOrder.erase(deleteRequest);

			if (status == tbricks::Status::FAIL) {
				TBWARNING("Cannot delete order: " << request->second);
			}
		}
		else {
				TBWARNING("Request NOT found in ORDERS OR DELETES: " << requestId);	
				return;
		}
	}
	if (status == tbricks::Status::FAIL) {
		OrderInfo& info = m_orders[request->second];
		tbricks::String reason = tbricks::String("Order failed with reason: " + statusText);
		TBWARNING(reason);

		OrderIdToOrderInfo::iterator o = m_orders.find(request->second);
		m_orders.erase(o);
		// In KSingleIVIDLeg create ILeg::Leg_HandleFailure
		// Make the parent strategy implement this interface
		// Pass the parent strategy when the KSingleIVIDLeg objects are created
		// Then you can call back into the parent from here in case of failure
	}
	 m_requestToOrder.erase(request);
	TBDEBUG("<KSingleIVIDLeg::HandleRequestReply()");
}


bool KSingleIVIDLeg::HandleOrderRecovery(const tbricks::Order::Update& update) 
{
	TBDEBUG(">KSingleIVIDLeg::HandleOrderRecovery()");// << update);
// KTK - may have resolution issues here if instrument and instrument trading information are not .resolved
	if (m_instrument.Empty() || m_ITI.Empty()) {
		TBDEBUG("----> Instrument or InstrumentTradingInformation is Empty()");
		return false;	
	}
	tbricks::InstrumentVenueIdentification ivid;
	if (update.GetInstrumentVenueIdentification(ivid) && (ivid.GetInstrumentIdentifier() == m_instrument.GetIdentifier())) {
		if (update.HasFilledVolume()) {
			tbricks::Volume filledVolume;
			update.GetFilledVolume(filledVolume);
			
			if (m_ITI.CompareVolumes(filledVolume.GetDouble(), 0) == 1) { 
				tbricks::Side side;	
				update.GetSide(side);
				
				if (side == tbricks::Side::BUY) {
					m_longVol += filledVolume;
				}
				else if (side == tbricks::Side::SELL) {
					m_shortVol += filledVolume;
				}
				TBDEBUG("----> Recovered: " << m_instrument.GetShortName() << " LongVol: " << m_longVol  << " ShortVol: " << m_shortVol);
			}
			return true;
		}
	}
	else {
		TBDEBUG("----> HandleOrderRecovery - ignoring (" << ivid << ") mine (" << m_ivid << ")");
		return false;
	}
	TBDEBUG("<KSingleIVIDLeg::HandleOrderRecovery()");
	return false;
}

/*
void KSingleIVIDLeg::HandleBestPrice(const StreamIdentifier& stream, const BestPrice& price) {

}

void KSingleIVIDLeg::HandleStreamOpen(const StreamIdentifier& stream) {

}

void KSingleIVIDLeg::HandleStreamClose(const StreamIdentifier& stream) {

}

void KSingleIVIDLeg::HandleStreamStale(const StreamIdentifier& stream) {

}

void KSingleIVIDLeg::HandleSnapshotEnd(const StreamIdentifier& stream) {

}
*/

void KSingleIVIDLeg::HandleTimerEvent(const tbricks::Timer& timer) 
{
	TBDEBUG(">KSingleIVIDLeg::HandleTimerEvent()" << timer);
	TBDEBUG("<KSingleIVIDLeg::HandleTimerEvent()" << timer);
}


void KSingleIVIDLeg::SendOrder(const tbricks::Volume& volume, const tbricks::Price& price, const tbricks::Side& side)
{
	TBDEBUG("-KSingleIVIDLeg::SendOrder()");
	tbricks::Order::Options options;	
	SendOrder(volume, price, side, options);
}

void KSingleIVIDLeg::SendOrder(const tbricks::Volume& volume, const tbricks::Price& price, const tbricks::Side& side, const tbricks::Order::Options& options)
{
	TBDEBUG("-KSingleIVIDLeg::SendOrder()");
	PrintQueueStats("0");
	// Don't send anything if we're not active - especially when we are trying to cancel all
	if (false == m_isActive) {
		return;
	}
	if (m_requestToOrder.size() > 0 || m_deleteRequestToOrder.size() > 0) {
		TBDEBUG("NO ORDER: There are pending requests for this leg - returning");
		return;
	}
	if (price.Empty() && volume.Empty()) {
		TBDEBUG("NO ORDER: Price and volume are empty");
		return;
	}

	
/*	
	for (OrderIdToOrderInfo::const_iterator i = m_orders.begin(); i != m_orders.end(); i++ ) {
	}	
*/
	bool shouldModifyPrice = true;
	bool shouldModifyVolume = true;
	PrintQueueStats("1");
	if (m_orders.size() ==  0) {
		OrderInfo info;
		tbricks::Order::Options orderOpts = options;
		orderOpts.SetPrice(price);
		orderOpts.SetSide(side);
		orderOpts.SetActiveVolume(volume);
		orderOpts.SetPortfolioIdentifier(m_portfolioId);
		orderOpts.SetInstrumentVenueIdentification(m_ivid);
		tbricks::OrderManager::OrderCreateRequestResult result = m_orderManager.SendCreateRequest(*this, orderOpts); 
		info.m_order = tbricks::Order(result.GetOrderIdentifier());
		info.m_options = orderOpts;
		info.m_pendingVolume = volume;
		info.m_pendingPrice = price;
		m_orders[result.GetOrderIdentifier()] = info;
		TBDEBUG("====> CREATE ORDER: " << result.GetOrderIdentifier());
		m_requestToOrder[result.GetRequestIdentifier()] = result.GetOrderIdentifier();
		TBDEBUG("====> MAP REQUEST: " << result.GetRequestIdentifier());
		PrintQueueStats("2");
	}
	else {
		PrintQueueStats("3");
		// KTK 20110411
		// There SHOULD be only one order in m_orders at a time
		// If there is an order and it has a price ans size then 
		// modify it. 
		// Note that we should really use VenueCapabilities here to see if we should 
		// cancel and then replace or we are allowed to send a modify directly
		for (OrderIdToOrderInfo::iterator i = m_orders.begin(); i != m_orders.end(); i++) {
			OrderInfo info;
			tbricks::Order::Modifier modifier;
			tbricks::Price currentPrice;
			tbricks::Volume currentVolume; 
			if (i->second.m_order.HasPrice()) {
				 i->second.m_order.GetPrice(currentPrice);
			}
			if (i->second.m_order.HasActiveVolume()) {
				 i->second.m_order.GetActiveVolume(currentVolume);
			}
			TBDEBUG("====> CHECKING REQUEST AGAINST: " << currentVolume << " @ " << currentPrice << " " << i->second.m_order.GetIdentifier());
			// Check prices
			if (i->second.m_order.GetPrice(currentPrice) && (m_ITI.GetTickRule().ComparePrices(currentPrice, price) == tbricks::TB_EQUAL))  {
				shouldModifyPrice = false;
				//TBDEBUG("Skipping price modification from: " << currentPrice << " to: " << price);
			}
			else if (!price.Empty()) {
				modifier.SetPrice(price);
			}
			// Check volumes
			if (i->second.m_order.GetActiveVolume(currentVolume) && m_ITI.CompareVolumes(currentVolume, volume) == tbricks::TB_EQUAL) {
				shouldModifyVolume = false;
				//TBDEBUG("Skipping volume modification from: " << currentVolume << " to: " << volume);
			}
			else if (!volume.Empty()) {
				modifier.SetActiveVolume(volume);
			}	
			if (shouldModifyVolume || shouldModifyPrice) {
				TBDEBUG("====> MODIFY ORDER: " << i->first);
				tbricks::Identifier requestId = m_orderManager.SendModifyRequest(*this, i->first, modifier);	
				i->second.m_pendingPrice = price;
				i->second.m_pendingVolume = volume;
				m_requestToOrder[requestId] = i->first;
			}
		}
		PrintQueueStats("4");
	}
}

void 
KSingleIVIDLeg::CancelAll()
{
	TBDEBUG(">KSingleIVIDLeg::CancelAll()");
	PrintQueueStats("FROM: KSingleIVIDLeg::CancelAll()");
	for (OrderIdToOrderInfo::const_iterator it = m_orders.begin(); it != m_orders.end(); it++) {
		TBDEBUG("====> DELETE ORDER: " << it->second.m_order.GetIdentifier());
		tbricks::Identifier requestId = m_orderManager.SendDeleteRequest(*this, it->second.m_order.GetIdentifier());
		m_deleteRequestToOrder[requestId] = it->second.m_order.GetIdentifier();
	}
	TBDEBUG("<KSingleIVIDLeg::CancelAll()");
}

size_t 
KSingleIVIDLeg::GetNumOrdersOnMarket() const
{
	size_t nOrders = 0;
	OrderIdToOrderInfo::const_iterator i = m_orders.begin();
	for (i = m_orders.begin(); i != m_orders.end(); i++) {
		if (m_orderManager.IsOrderOnMarket(i->second.m_order)) {
			nOrders++;
		}
	}
	return nOrders;
}

bool 
KSingleIVIDLeg::IsOrderOnMarket(const tbricks::Order& o) const 
{
	return m_orderManager.IsOrderOnMarket(o);
}

bool
KSingleIVIDLeg::IsOrderOnMarket(const tbricks::OrderIdentifier& oi) const
{
	if (!oi.Empty()) {
		tbricks::Order o(oi);
		return m_orderManager.IsOrderOnMarket(o);
	}
	return false;	
}

void 
KSingleIVIDLeg::PrintQueueStats(const tbricks::String& string)
{
	TBDEBUG(">KSingleIVIDLeg::PrintQueueStats()");
	TBDEBUG(string << " - Requests: "  << m_requestToOrder.size() << ", Orders:   "  << m_orders.size() << ", Deletes:  "  << m_deleteRequestToOrder.size());	
	TBDEBUG("<KSingleIVIDLeg::PrintQueueStats()");
}
