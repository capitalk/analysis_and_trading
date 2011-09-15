/* 
 * File:   Info.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 5:09 PM
 */

#ifndef _INFO_HXX
#define	_INFO_HXX

#include <strategy/API.h>

namespace AOStatistics
{

struct Info
{
   enum Side { BID = 0, ASK, NONE, TOTAL, SIDE_MAX };

   tbricks::TickRule _rule;
   unsigned int _countTrades[SIDE_MAX];
   tbricks::Volume _countVolume[SIDE_MAX];
   tbricks::Price _lastPrice[SIDE_MAX];
   tbricks::Price _bestPrice[SIDE_MAX];

   void reset()
   {
      for(size_t i = 0; i < SIDE_MAX; ++i)
      {
         _countTrades[i] = 0;
         _countVolume[i] = 0;
         _lastPrice[i].Clear();
         _bestPrice[i].Clear();
      }
   }

   Info::Side getTradeSide(const tbricks::Price& price) const
   {
      if (_bestPrice[ASK].Empty() == false && _rule.ComparePrices(price, _bestPrice[ASK]) != tbricks::TB_LESS_THAN)
         return ASK;
      else if (_bestPrice[BID].Empty() == false && _rule.ComparePrices(price, _bestPrice[BID]) != tbricks::TB_GREATER_THAN)
         return BID;
      else
         return NONE;
   }

   tbricks::Volume getVolume(const Info::Side& side = Info::TOTAL) const
   {
      if (side == Info::TOTAL)
         return _countVolume[BID] + _countVolume[ASK] + _countVolume[NONE];
      else
         return _countVolume[side];
   }

   unsigned int getCount(const Info::Side& side = Info::TOTAL) const
   {
      if (side == Info::TOTAL)
         return _countTrades[BID] + _countTrades[ASK] + _countTrades[NONE];
      else
         return _countTrades[side];
   }

   tbricks::Double getVolumePercentage(const Info::Side& side) const
   {
      tbricks::Volume totVolume = getVolume(Info::TOTAL);
      if(totVolume > 1e-5)
         return getVolume(side).GetDouble() / totVolume;
      else
         return tbricks::Double();
   }

   tbricks::Double getCountPercentage(const Info::Side& side) const
   {
      unsigned int totTrades = getCount(Info::TOTAL);
      if(totTrades > 0)
         return getCount(side) / tbricks::Double(totTrades);
      else
         return tbricks::Double();
   }
};

}

#endif	/* _INFO_HXX */

