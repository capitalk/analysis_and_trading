/*
 *  InstrumentPosition.h
 *  Strategies
 *
 *  Created by Konstantin Romanov on 8/13/09.
 *  Copyright 2009-2011 Tbricks. All rights reserved.
 *
 */

#ifndef InstrumentPosition_H
#define InstrumentPosition_H

#include <strategy/API.h>

using namespace tbricks;

class InstrumentPosition : protected BestPriceStream::IHandler, 
                           protected StatisticsStream::IHandler 

{
public:
  class IModuleHandler
    {
    public:
      virtual void InstrumentPosition_PnLChanged(InstrumentPosition& position) = 0;
    };
  
  InstrumentPosition(const InstrumentVenueIdentification& ivid, InstrumentPosition::IModuleHandler& handler);
  virtual ~InstrumentPosition();
  
  bool UpdateFromPosition(const Position& position);
  bool InvalidatePosition(const Identifier& positionId);
  
  const Double& GetNetPosition() const { return m_netPosition; }
  const Double&  GetRealizedPnL() const { return m_realizedPnL; }
  const Double&  GetUnrealizedPnL() const { return m_unrealizedPnL; }
  const Double& GetTurnover() const { return m_turnover; }
  
  const Instrument& GetInstrument() const { return m_instrument; }

  Price GetReferencePrice(bool isLongPosition) const;

protected:
  virtual void HandleBestPrice(const StreamIdentifier & stream, const BestPrice & price);
  virtual void HandleStatistics(const StreamIdentifier & stream, const Statistics & statistics);
  
  virtual void HandleStreamOpen(const StreamIdentifier & stream);
  virtual void HandleStreamStale(const StreamIdentifier & stream);
  virtual void HandleStreamClose(const StreamIdentifier & stream);
  virtual void HandleSnapshotEnd(const StreamIdentifier & stream);
  
private:
  bool RecalculateUnrealizedPnL();
  
  IModuleHandler&   m_handler;
  Instrument        m_instrument;
  Double            m_netPosition;
  Double            m_realizedPnL;
  Double            m_unrealizedPnL;
  Double            m_turnover;
  BestPriceStream   m_bestPriceStream;
  StatisticsStream  m_statisticsStream;
  Price             m_bid;
  Price             m_ask;
  Price             m_last;
  Price             m_close;
  
  typedef std::map<Identifier, Position>  Positions;
  Positions         m_positions;
};

#endif // InstrumentPosition_H


