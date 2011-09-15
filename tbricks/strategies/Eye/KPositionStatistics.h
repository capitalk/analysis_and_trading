/* 
 * Copyright 2011 Capital K Partners BV 
 * Timir Karia
 */

/*  Position statistics
 *  1) Keep track of position changes for a single INSTRUMENT (maybe on multiple markets) 
 *  2) Compute summary statistics for volume and prices
 *  3) Record last trade price, size, side, MIC TODO - should probably record Venue as well
 *  4) Allow user to request a hedgeable position and price based on the position that is currently set 
 *  5) Track total turnover on a MIC basis
 *  6) Be prinatable to logs 
 */



#include "strategy/API.h"
#include "ManagedOrderModule.h"
#include "Utils.h"

#ifndef __POSITIONSTATISTICS__
#define __POSITIONSTATISTICS__


class KPositionStatistics: protected tbricks::Printable
{
	public: 
		KPositionStatistics();
		KPositionStatistics(const tbricks::InstrumentIdentifier& instrument);
		void Update(const tbricks::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const tbricks::MIC& mic, const tbricks::InstrumentVenueIdentification& ivid);
		inline bool isFlat() { return (m_longVolume == m_shortVolume);}
		void getLongVolume(tbricks::Volume & v) const { v = m_longVolume;}	
		void getShortVolume(tbricks::Volume & v) const { v =  m_shortVolume;}	
		void getLongPaid(tbricks::Price & p ) const { p = m_longPaid;}
		void getShortPaid(tbricks::Price & p) const { p = m_shortPaid;}
		void getShortWAvgPaid(tbricks::Price & p) const { p =  m_shortWAvgPaid;}
		void getLongWAvgPaid(tbricks::Price & p) const { p = m_longWAvgPaid;}
		void getLastTradeMIC(tbricks::MIC & m) const { m = m_lastTradeMic;}
		void getLastTradeIVID(tbricks::InstrumentVenueIdentification & ivid) const { ivid = m_lastTradeIvid;}
		void getLastTradeVolume(tbricks::Volume & v) const { v = m_lastTradeVolume;}
		void getLastTradePrice(tbricks::Price & p) const { p = m_lastTradePrice;}
		void getLastTradeSide(tbricks::Side & s) const { s = m_lastTradeSide;}
		void setInstrument(const tbricks::Instrument& instrument);
		void setInstrumentId(const tbricks::InstrumentIdentifier & iid);
		void getInstrumentId(tbricks::InstrumentIdentifier & iid) const;
		void getHedgeSideAndVolume(tbricks::Side & hedgeSide, tbricks::Volume & hedgeVolume) const;

		virtual std::ostream& Print(std::ostream& strm) const;
		virtual tbricks::String ToString() const;

	protected:
		tbricks::Volume m_longVolume;
		tbricks::Volume m_shortVolume;
		tbricks::Price m_longPaid;
		tbricks::Price m_shortPaid;
		tbricks::Price m_longWAvgPaid;
		tbricks::Price m_shortWAvgPaid;
		tbricks::MIC m_lastTradeMic;
		tbricks::InstrumentVenueIdentification m_lastTradeIvid;
		tbricks::Volume m_lastTradeVolume;
		tbricks::Price m_lastTradePrice;
		tbricks::Side m_lastTradeSide;
	private:
		tbricks::InstrumentIdentifier m_instrumentId;
		size_t m_nTrades;

		tbricks::Hash<tbricks::MIC, tbricks::Volume> m_micToTurnover;
		STRATEGY_EXPORT friend std::ostream& operator<<(std::ostream &strm, const tbricks::Printable &printable);
};

#endif // __POSITIONSTATISTICS__
