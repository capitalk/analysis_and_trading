#include "WindowContainer.hxx"

namespace AOStatistics
{

WindowContainer::WindowContainer(const double& timeWindow) : Container(timeWindow),
   _timeWindow(timeWindow)
{

}

WindowContainer::~WindowContainer()
{

}

void WindowContainer::start()
{
   _historyTrades.clear();
   Container::start();
}

void WindowContainer::handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time, Info::Side side)
{
   //first 'undo' all trades that get out of scope (=out of the sliding window)
   while(_historyTrades.empty() == false && (time - _historyTrades.front().get < 3 > ()) > _timeWindow)
   {
      const Info::Side& undoSide = _historyTrades.front().get < 0 >();
      const tbricks::Volume& undoVolume = _historyTrades.front().get < 1 >();
      const tbricks::Price& undoPrice = _historyTrades.front().get < 2 >();
      const double& undoTime = _historyTrades.front().get < 3 >();
      for (ContStats::iterator it = _stats.begin(); it != _stats.end(); ++it)
         (*it)->undoTrade(undoVolume, undoPrice, undoTime, undoSide);
         
      --_helper._countTrades[undoSide];
      _helper._countVolume[undoSide] -= undoVolume;
//      TBNOTICE("UNDO Volume " << undoVolume << " traded on " << side << " @ " << undoTime << " - New volume count: " <<
//         _helper._countVolume[Info::BID] << ", " << _helper._countVolume[Info::ASK] << ", " << _helper._countVolume[Info::NONE]);
      
      _historyTrades.pop_front();
   }
   finalizeUndo(_historyTrades);

   //find the side of the new trade side (if not already specified)
   if(side >= Info::NONE)
      side = _helper.getTradeSide(price);

   //keep the trade in our history container
   _historyTrades.push_back(boost::make_tuple(side, volume, price, time));

   //do the default handling (= keep basic stats and call all components)
   Container::handleTrade(volume, price, time, side);
}

const ContHistoryTrades WindowContainer::getActiveTrades() const
{
   return _historyTrades;
}

}
