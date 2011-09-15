/* 
 * File:   MarketPhases.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on April 21, 2010, 5:47 PM
 */

#ifndef _MARKETPHASES_HXX
#define	_MARKETPHASES_HXX

#include "strategy/API.h"
#include <vector>

namespace common
{

class MarketPhases
{
public:

   enum MarketPhase
   {
      NONE = 0, TRADING = 1, AUCTION = 2, LAST = 3, CLOSED = 4, HALTED = 5, DISABLED = -1
   };

   typedef std::vector< std::pair< tbricks::DateTime, int > > ContTimePairs;
   static MarketPhase getMarketPhase(const tbricks::String& venueName, const tbricks::String& venueInstrumentTradingStatus);
   static void getTimeBasedChanges(ContTimePairs &timePairs, const tbricks::String& venueName, const int& index = -1);
   static const char* getPhaseDescription(const MarketPhase& marketPhase);

private:

   struct HashMarketPhase
   {

      HashMarketPhase()
      {
         m_table["Euronext"]["A"] = AUCTION;
         m_table["Euronext"]["H"] = HALTED;
         m_table["Euronext"]["C"] = CLOSED;
         m_table["Euronext"]["EAMO"] = NONE;
         m_table["Euronext"]["COCA"] = AUCTION;
         m_table["Euronext"]["COAU"] = AUCTION;
         m_table["Euronext"]["COCO"] = TRADING;
         m_table["Euronext"]["CLCA"] = AUCTION;
         m_table["Euronext"]["CLAU"] = AUCTION;
         m_table["Euronext"]["TAL"] = LAST;
         m_table["Euronext"]["COMO"] = CLOSED;
         m_table["Euronext"]["LAMO"] = CLOSED;
         m_table["Euronext"]["HALT"] = HALTED;
         m_table["Euronext"]["Unknown"] = AUCTION;

         m_table["Xetra"]["Start"] = CLOSED;
         m_table["Xetra"]["Pre Trading"] = CLOSED;
         m_table["Xetra"]["Pre-call"] = CLOSED;
         m_table["Xetra"]["Crossing Period"] = HALTED;
         m_table["Xetra"]["Closing Crossing Period"] = HALTED;
         m_table["Xetra"]["Opening Auction Call"] = AUCTION;
         m_table["Xetra"]["Intra Day Auction Call"] = AUCTION;
         m_table["Xetra"]["Closing Auction Call"] = AUCTION;
         m_table["Xetra"]["End Auction Call"] = AUCTION;
         m_table["Xetra"]["Auction Call"] = AUCTION;
         m_table["Xetra"]["Opening Auction IPO Call"] = AUCTION;
         m_table["Xetra"]["Opening Auction IPO Freeze"] = HALTED;
         m_table["Xetra"]["Intra Day Auction IPO Call"] = AUCTION;
         m_table["Xetra"]["Intra Day Auction IPO Freeze"] = HALTED;
         m_table["Xetra"]["IPO"] = NONE;
         m_table["Xetra"]["Quote Driven IPO Freeze"] = HALTED;
         m_table["Xetra"]["Opening Auction Pre-Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Intra Day Auction Pre-Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Closing Auction Pre-Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["End-of-day Auction Pre-Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Pre-Orderbook Balancing of quote driver auction"] = AUCTION;
         m_table["Xetra"]["Opening Auction Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Intra Day Auction Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Closing Auction Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["End-of-day Auction Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Orderbook Balancing"] = AUCTION;
         m_table["Xetra"]["Continuous Trading"] = TRADING;
         m_table["Xetra"]["In Between Auctions"] = NONE;
         m_table["Xetra"]["Post Trading"] = CLOSED;
         m_table["Xetra"]["End Of Trading"] = CLOSED;
         m_table["Xetra"]["Halt"] = HALTED;
         m_table["Xetra"]["Suspend"] = HALTED;
         m_table["Xetra"]["Volatility Interruption"] = AUCTION;
         m_table["Xetra"]["Add"] = NONE;
         m_table["Xetra"]["Delete"] = NONE;
         m_table["Xetra"]["Call Unfreeze"] = HALTED;
         m_table["Xetra"]["Continuous Auction Pre-Call"] = AUCTION;
         m_table["Xetra"]["Continuous Auction Call"] = AUCTION;
         m_table["Xetra"]["Continuous Auction Freeze"] = HALTED;

         m_table["Eurex"]["a"] = AUCTION;
         m_table["Eurex"]["A"] = AUCTION;
         m_table["Eurex"]["c"] = CLOSED;
         m_table["Eurex"]["C"] = CLOSED;
         m_table["Eurex"]["f"] = HALTED;
         m_table["Eurex"]["F"] = HALTED;
         m_table["Eurex"]["h"] = HALTED;
         m_table["Eurex"]["H"] = HALTED;
         m_table["Eurex"]["l"] = CLOSED;
         m_table["Eurex"]["L"] = CLOSED;
         m_table["Eurex"]["n"] = CLOSED;
         m_table["Eurex"]["o"] = AUCTION;
         m_table["Eurex"]["O"] = AUCTION;
         m_table["Eurex"]["p"] = AUCTION;
         m_table["Eurex"]["P"] = AUCTION;
         m_table["Eurex"]["r"] = NONE;
         m_table["Eurex"]["R"] = NONE;
         m_table["Eurex"]["v"] = TRADING;
         m_table["Eurex"]["V"] = TRADING;
         m_table["Eurex"]["X"] = HALTED;
         m_table["Eurex"]["x"] = HALTED;
         m_table["Eurex"][" "] = TRADING; // space
         m_table["Eurex"]["I"] = HALTED;
         m_table["Eurex"]["i"] = HALTED;
         m_table["Eurex"]["START"] = CLOSED;
         m_table["Eurex"]["PRETR"] = AUCTION;
         m_table["Eurex"]["PREOP"] = AUCTION;
         m_table["Eurex"]["FREEZ"] = HALTED;
         m_table["Eurex"]["TRAD"] = TRADING;
         m_table["Eurex"]["CLAUC"] = AUCTION;
         m_table["Eurex"]["HALT"] = HALTED;
         m_table["Eurex"]["POSTF"] = CLOSED;
         m_table["Eurex"]["POSTR"] = CLOSED;
         m_table["Eurex"]["POST1"] = CLOSED;
         m_table["Eurex"]["POST2"] = CLOSED;
         m_table["Eurex"]["BATCH"] = CLOSED;
         m_table["Eurex"]["FAST"] = TRADING;
         m_table["Eurex"]["EXCNT"] = CLOSED;
         m_table["Eurex"]["HOLID"] = CLOSED;
         m_table["Eurex"]["ONLIN"] = NONE;
         m_table["Eurex"]["INACT"] = HALTED;

         m_table["NASDAQ OMX INET Nordic"]["Pre-open"] = CLOSED;
         m_table["NASDAQ OMX INET Nordic"]["Pre-Open"] = CLOSED;
         m_table["NASDAQ OMX INET Nordic"]["Opening Auction"] = AUCTION;
         m_table["NASDAQ OMX INET Nordic"]["Continuous Trading"] = TRADING;
         m_table["NASDAQ OMX INET Nordic"]["Closing Auction"] = AUCTION;
         m_table["NASDAQ OMX INET Nordic"]["Halted"] = HALTED;
         m_table["NASDAQ OMX INET Nordic"]["Post-Trade"] = CLOSED;
         m_table["NASDAQ OMX INET Nordic"]["Closed"] = CLOSED;

         m_table["SWXess"]["0 - Delayed Opening"] = NONE;
         m_table["SWXess"]["1 - Delayed Opening with Non Opening"] = NONE;
         m_table["SWXess"]["2 - Non Opening"] = NONE;
         m_table["SWXess"]["3 - None"] = AUCTION;
         m_table["SWXess"]["4 - Stop Trading"] = HALTED;
         m_table["SWXess"]["5 - Stop Trading with Non Opening"] = HALTED;
         m_table["SWXess"]["5 - Underlying Condition"] = NONE;
         m_table["SWXess"]["5 - Underlying Condition with Non Opening"] = NONE;

         m_table["SWXess"]["Pre-Open | None"] = AUCTION;
         m_table["SWXess"]["Open | None"] = TRADING;
         m_table["SWXess"]["Closing Auction | None"] = AUCTION;
         m_table["SWXess"]["Auction Close | None"] = CLOSED;

         m_table["Baxter FX"]["Open/"] = TRADING;

      }
      tbricks::Hash<tbricks::String, tbricks::Hash<tbricks::String, MarketPhase> > m_table;
   };
   static HashMarketPhase m_phaseLookup;
};

}

#endif	/* _MARKETPHASES_HXX */

