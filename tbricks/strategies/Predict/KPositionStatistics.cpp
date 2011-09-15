#include "KPositionStatistics.h"

KPositionStatistics::KPositionStatistics(const tbricks::InstrumentIdentifier& instrumentId):m_longVolume(0), m_shortVolume(0), m_longPaid(0), m_shortPaid(0), m_longWAvgPaid(0), m_shortWAvgPaid(0), m_instrumentId(instrumentId), m_nTrades(0)
{

}

void 
KPositionStatistics::Update(const tbricks::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const tbricks::MIC& mic, const tbricks::InstrumentVenueIdentification& ivid)
{
	m_nTrades++;
	if (side == tbricks::Side::BUY) {
		m_longVolume += volume;
		m_longPaid += (volume*price);
		m_longWAvgPaid = (m_longVolume / m_longPaid);
	}
	if (side == tbricks::Side::SELL) {
		m_shortVolume += volume;
		m_shortPaid += (volume*price);
		m_shortWAvgPaid = (m_shortVolume / m_shortPaid);
	}
	m_micToTurnover[mic] += volume;
	m_lastTradePrice = price;
	m_lastTradeVolume = volume;
	m_lastTradeMic = mic;	
	m_lastTradeIvid = ivid;	
	m_lastTradeSide = side;	
}

void 
KPositionStatistics::getInstrumentId(tbricks::InstrumentIdentifier & iid) const
{
	iid = m_instrumentId;
}

void 
KPositionStatistics::setInstrumentId(const tbricks::InstrumentIdentifier & iid) 
{
	if (iid.Empty()) {
		return;
	}
	m_instrumentId = iid;
}

void 
KPositionStatistics::setInstrument(const tbricks::Instrument & i) 
{
	if (i.Empty()) {
		return;
	}
	m_instrumentId = i.GetIdentifier();
}

void
KPositionStatistics::getHedgeSideAndVolume(tbricks::Side& hedgeSide, tbricks::Volume& hedgeVolume) const
{
	tbricks::Volume v; 
	if (m_longVolume > m_shortVolume) {
		hedgeVolume = (m_longVolume - m_shortVolume);
		hedgeSide = tbricks::Side::SELL;
	}
	if (m_shortVolume > m_longVolume) { 
		hedgeVolume = (m_shortVolume - m_longVolume);
		hedgeSide = tbricks::Side::BUY;
	}
} 

std::ostream&
KPositionStatistics::Print(std::ostream& strm) const
{
	return (strm << ToString());	
}
			
tbricks::String
KPositionStatistics::ToString() const
{
	tbricks::String s;
	tbricks::Volume hv;
	tbricks::Side hs;	
	getHedgeSideAndVolume(hs, hv);

	s += "\tLongVolume: " +  m_longVolume.ToString() + " LongPaid: " + m_longPaid.ToString()  + " LongWAvgPaid: " + m_longWAvgPaid.ToString() \
	+ "\n" \
		+ "\tShortVolume: " + m_shortVolume.ToString()  + " ShortPaid: " + m_shortPaid.ToString()  + " ShortWAvgPaid: " + m_shortWAvgPaid.ToString() \
		+ "\tTO HEDGE:\n" + hs.ToString() + " " + hv.ToString(); 
	s+= "\n";
		tbricks::Hash<tbricks::MIC, tbricks::Volume>::const_iterator it; 
		tbricks::MIC m;
		for(it = m_micToTurnover.begin(); it != m_micToTurnover.end(); it++) {
			m = it->first;
			//s += it->first.GetMIC();
			s += " : ";
			s += it->second.ToString();	
		}
		return s;
}

