/* 
 * File:   WindowContainer.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:05 PM
 */

#ifndef _STAT_WindowContainer_HXX
#define	_STAT_WindowContainer_HXX

#include "Container.hxx"
#include <strategy/API.h>
#include <list>
#include <boost/tuple/tuple.hpp>

namespace AOStatistics
{

//typedef std::list< boost::tuple<Info::Side, tbricks::Volume, tbricks::Price, double> > ContHistoryTrades;  //side, volume, price, time

class WindowContainer : public Container
{
public:
   WindowContainer(const double& timeWindow);
   virtual ~WindowContainer();

   virtual void start();
   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE);

   const ContHistoryTrades getActiveTrades() const;

protected:
   const double _timeWindow;
   ContHistoryTrades _historyTrades;
};

}
#endif	/* _WindowContainer_HXX */

