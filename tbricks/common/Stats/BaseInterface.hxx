/* 
 * File:   BaseInterface.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:09 PM
 */

#ifndef _STAT_BASEINTERFACE_HXX
#define	_STAT_BASEINTERFACE_HXX

#include "Info.hxx"
#include <strategy/API.h>
#include <stdexcept>

#include <list>
#include <boost/tuple/tuple.hpp>

namespace AOStatistics
{

typedef std::list< boost::tuple<Info::Side, tbricks::Volume, tbricks::Price, double> > ContHistoryTrades;  //side, volume, price, time

class BaseInterface
{
public:

   BaseInterface(const Info& helper) :
   _refHelper(helper)
   {
   }

   virtual ~BaseInterface()
   {
   }

   virtual void start() = 0;
   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE) = 0;
   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0) = 0;
   virtual void undoTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      throw new std::runtime_error("Statistics components does not support sliding window!");
   }
   virtual void finalizeUndo(const ContHistoryTrades& trades)
   {

   }

protected:
   const Info& _refHelper;
};

}

#endif	/* _BASEINTERFACE_HXX */

