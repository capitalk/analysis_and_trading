/* 
 * File:   Bar.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on July 21, 2010, 16:05 PM
 */

#ifndef _STAT_Bar_HXX
#define	_STAT_Bar_HXX

#include "BaseInterface.hxx"
#include "WindowContainer.hxx"

namespace AOStatistics
{

class Bar : public BaseInterface
{
public:

   Bar(const Info& helper, const Info::Side& filterSide = Info::TOTAL) :
      BaseInterface(helper),
      _filterSide(filterSide)
   {

   }

   virtual ~Bar()
   {

   }

   virtual void start()
   {
      _open.Clear();
      _close.Clear();
      _low.Clear();
      _high.Clear();
   }

   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_filterSide == Info::TOTAL || _filterSide == side)
      {
         _close = price;
         if(_open.Empty())
            _open = price;
         if(_low.Empty() || _low > price)
            _low = price;
         if(_high.Empty() || _high < price)
            _high = price;
      }
   }

   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0)
   {
   }

   virtual void undoTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_filterSide == Info::TOTAL || _filterSide == side)
      {
         _open.Clear();
         if(_refHelper._rule.ComparePrices(price, _low) == tbricks::TB_EQUAL)
            _low.Clear();
         if(_refHelper._rule.ComparePrices(price, _high) == tbricks::TB_EQUAL)
            _high.Clear();
      }
   }

   virtual void finalizeUndo(const ContHistoryTrades& trades)
   {
      if(trades.empty())
      {
         _close.Clear();
         return;
      }

      tbricks::Price low = _low;
      tbricks::Price high = _high;
      for(ContHistoryTrades::const_iterator it = trades.begin(); it != trades.end() &&
         (_open.Empty() || _low.Empty() || _high.Empty()); ++it)
      {
         if (_filterSide == Info::TOTAL || _filterSide == it->get < 0 >())
         {
            const tbricks::Price& price = it->get < 2 >();
            if(_open.Empty())
               _open = price;
            if(_low.Empty() && (low.Empty() || low > price))
               low = price;
            if(_high.Empty() && (high.Empty() || high < price))
               high = price;
         }
      }

      if(_low.Empty())
         _low = low;
      if(_high.Empty())
         _high = high;
   }

   const tbricks::Price& getOpen() const
   {
      return _open;
   }

   const tbricks::Price& getClose() const
   {
      return _close;
   }

   const tbricks::Price& getLow() const
   {
      return _low;
   }

   const tbricks::Price& getHigh() const
   {
      return _high;
   }

protected:
   tbricks::Price _open;
   tbricks::Price _close;
   tbricks::Price _low;
   tbricks::Price _high;
   Info::Side _filterSide;
};

}

#endif	/* _STATBar_HXX */

