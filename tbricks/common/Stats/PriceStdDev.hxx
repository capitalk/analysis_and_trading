/* 
 * File:   PriceStdDev.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:05 PM
 */

#ifndef _STAT_PriceStdDev_HXX
#define	_STAT_PriceStdDev_HXX

#include "BaseInterface.hxx"
#include "WindowContainer.hxx"
#include "boost/accumulators/accumulators.hpp"
#include "boost/accumulators/statistics.hpp"
#include <boost/accumulators/statistics/variance.hpp>

namespace AOStatistics
{

class PriceStdDev : public BaseInterface
{
public:

   //call this constructor if NOT used in a sliding window statistics container
   PriceStdDev(const Info& helper, const Info::Side& filterSide = Info::TOTAL) :
      BaseInterface(helper),
      _filterSide(filterSide),
      _refHistoryTrades(_dummyHistoryTrades)
   {

   }

   //call this constructor if used in a sliding window statistics container, supplying a reference to WindowContainer::getActiveTrades()
   PriceStdDev(const Info& helper, const ContHistoryTrades& historyTrades, const Info::Side& filterSide = Info::TOTAL) :
      BaseInterface(helper),
      _filterSide(filterSide),
      _refHistoryTrades(historyTrades)
   {

   }

   virtual ~PriceStdDev()
   {

   }

   virtual void start()
   {
      _acc = boost::accumulators::accumulator_set< double, boost::accumulators::stats <
         boost::accumulators::tag::variance(boost::accumulators::immediate) > >();
   }

   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_refHistoryTrades.empty() && (_filterSide == Info::TOTAL || _filterSide == side))
      {
         _acc(price.GetValue());
      }
   }

   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0)
   {
   }

   virtual void undoTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      //shouldn't get here if the _refHistoryTrades is not supplied (->will contain items)
      if(_refHistoryTrades.empty())
         BaseInterface::undoTrade(volume, price, time, side);
   }

   tbricks::Double getPriceStdDev() const
   {
      if (_refHistoryTrades.empty() == false)
      {
         _acc = boost::accumulators::accumulator_set< double, boost::accumulators::stats <
            boost::accumulators::tag::variance(boost::accumulators::immediate) > >();
         for (ContHistoryTrades::iterator it = _refHistoryTrades.begin(); it != _refHistoryTrades.end(); ++it)
            if (_filterSide == Info::TOTAL || _filterSide == it->get < 0 > ())
               _acc(it->get < 2 > ().GetValue());
      }

      double variance = boost::accumulators::variance(_acc);
      if(std::abs(variance) > 1e-6)
         return std::sqrt(variance);
      else
         tbricks::Double();
   }

protected:
   double _PriceStdDevSum;
   Info::Side _filterSide;
   const ContHistoryTrades& _refHistoryTrades;
   const ContHistoryTrades _dummyHistoryTrades;
   boost::accumulators::accumulator_set< double, boost::accumulators::stats<
      boost::accumulators::tag::variance(boost::accumulators::immediate) > > _acc;
};

}

#endif	/* _STATPriceStdDev_HXX */

