/*
 *  InstrumentPosition.cpp
 *  Strategies
 *
 *  Created by Konstantin Romanov on 8/13/09.
 *  Copyright 2009-2011 Tbricks. All rights reserved.
 *
 */

#include "InstrumentPosition.h"

InstrumentPosition::InstrumentPosition(const InstrumentVenueIdentification& ivid, InstrumentPosition::IModuleHandler& handler)
:m_handler(handler),
m_instrument(ivid.GetInstrumentIdentifier()),
m_bestPriceStream(*this),
m_statisticsStream(*this)
{
  TBDEBUG("PnLCalculator::InstrumentPosition::InstrumentPosition() for " << m_instrument.GetShortName());
  
  m_bestPriceStream.SuppressVolumeUpdates(true);
  m_bestPriceStream.SetPriority(1);
  m_bestPriceStream.Open(Stream::SNAPSHOT_AND_LIVE, ivid);
  
  m_statisticsStream.SetPriority(1);
  m_statisticsStream.Open(Stream::SNAPSHOT_AND_LIVE, ivid);
}

InstrumentPosition::~InstrumentPosition()
{
  TBDEBUG("InstrumentPosition::~InstrumentPosition() for " << m_instrument.GetShortName());
}

bool InstrumentPosition::InvalidatePosition(const Identifier& positionId)
{
  Positions::iterator i = m_positions.find(positionId);
  if (i == m_positions.end())
  {
    return false;
  }

  Position removedPosition = i->second;
  Volume positionVolume = removedPosition.GetVolumeBought() - removedPosition.GetVolumeSold();

  m_netPosition -= positionVolume;
  m_realizedPnL += removedPosition.GetGrossTradeAmountSold() - removedPosition.GetGrossTradeAmountBought();
  m_turnover -= removedPosition.GetGrossTradeAmountSold() + removedPosition.GetGrossTradeAmountBought();

  m_positions.erase(i);
  RecalculateUnrealizedPnL();

  return true;
}

bool InstrumentPosition::UpdateFromPosition(const Position& position)
{
  Positions::iterator i = m_positions.find(position.GetIdentifier());
  if (i == m_positions.end())
  {
    Volume addedVolume = position.GetVolumeBought() - position.GetVolumeSold();
    if (m_netPosition.Empty())
    {
      m_netPosition = Volume(0);
      m_realizedPnL = Double(0);
      m_turnover = Double(0);
    }
    m_netPosition += addedVolume;
    m_realizedPnL += position.GetGrossTradeAmountSold() - position.GetGrossTradeAmountBought();
    m_turnover += position.GetGrossTradeAmountSold() + position.GetGrossTradeAmountBought();

    m_positions[position.GetIdentifier()] = position;
  }
  else    
  {
    Volume addedVolume = (position.GetVolumeBought() - position.GetVolumeSold()) - (i->second.GetVolumeBought() - i->second.GetVolumeSold());
    Double addedCash = (position.GetGrossTradeAmountSold() - position.GetGrossTradeAmountBought()) - (i->second.GetGrossTradeAmountSold() - i->second.GetGrossTradeAmountBought());
    Double addedTurnover = (position.GetGrossTradeAmountSold() + position.GetGrossTradeAmountBought()) - (i->second.GetGrossTradeAmountSold() + i->second.GetGrossTradeAmountBought());
    
    m_netPosition += addedVolume;
    m_realizedPnL += addedCash;
    m_turnover += addedTurnover;
    
    i->second = position;
  }

  RecalculateUnrealizedPnL();
  
  return true;
}

bool InstrumentPosition::RecalculateUnrealizedPnL()
{
  Double newUnrealizedPnL;

  if (!m_netPosition.Empty())
  {
    Price referencePrice = GetReferencePrice(m_netPosition > 0);

    if (!referencePrice.Empty())
    {
      m_unrealizedPnL = m_netPosition * referencePrice;
    }
  }

  if (newUnrealizedPnL != m_unrealizedPnL)
  {
    m_unrealizedPnL = newUnrealizedPnL;
    return true;
  }
  else
  {
    return false;
  }
}


void InstrumentPosition::HandleBestPrice(const StreamIdentifier & stream, const BestPrice & price)
{
  if (price.BidPriceUpdated())
  {
    m_bid = price.GetBidPrice();
  }
  if (price.AskPriceUpdated())
  {
    m_ask = price.GetAskPrice();
  }
  if (RecalculateUnrealizedPnL())
  {
    m_handler.InstrumentPosition_PnLChanged(*this);
  }
}

void InstrumentPosition::HandleStatistics(const StreamIdentifier & stream, const Statistics & statistics)
{
  bool shouldRecalc = false;

  if (statistics.HasLastPrice())
  {
    m_last = statistics.GetLastPrice();
    shouldRecalc = true;
  }
  if (statistics.HasClosingPrice())
  {
    m_close = statistics.GetClosingPrice();
    shouldRecalc = true;
  }

  if (shouldRecalc && RecalculateUnrealizedPnL())
  {
    m_handler.InstrumentPosition_PnLChanged(*this);
  }
}

void InstrumentPosition::HandleStreamOpen(const StreamIdentifier & stream)
{
}

void InstrumentPosition::HandleStreamStale(const StreamIdentifier & stream)
{
}

void InstrumentPosition::HandleStreamClose(const StreamIdentifier & stream)
{
  TBSTATUS("Instrument position for " << m_instrument.GetShortName() << " - stream closed.");
}

void InstrumentPosition::HandleSnapshotEnd(const StreamIdentifier & stream)
{
}

Price InstrumentPosition::GetReferencePrice(bool isLongPosition) const
{
  Price referencePrice;
  
  if (isLongPosition)
  {
    // we're long, let's estimate on offer side
    referencePrice = m_ask;
  }
  else
  {
    // we're short - using bid
    referencePrice = m_bid;
  }
  
  if (referencePrice.Empty())
  {
    // no BBO -> try last, then close
    if (!m_last.Empty())
    {
      return m_last;
    }
    else
    {
      return m_close;
    }
  }
  
  return referencePrice;
}

