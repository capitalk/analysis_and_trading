#include "MarketPhases.hxx"

namespace common
{

using namespace tbricks;

MarketPhases::HashMarketPhase MarketPhases::m_phaseLookup;

MarketPhases::MarketPhase MarketPhases::getMarketPhase(const tbricks::String& venueName, const tbricks::String& venueInstrumentTradingStatus)
{  
   MarketPhase marketPhase = NONE; //by default
   if (venueName == "AllOptions")
   {
      marketPhase = TRADING;
   }
   else if (venueName == "Chi-X" || venueName == "BATS" || venueName == "neuro")
   {
      Double secsFromMidnight = DateTime::Now().Difference(DateTime::Today()).GetDouble();
      if (secsFromMidnight <= 9 * 3600 + 0 * 60 + 5 || secsFromMidnight >= 17 * 3600 + 29 * 60 + 58)
         marketPhase = CLOSED;
      else
         marketPhase = TRADING;
   }
   else if (marketPhase == NONE && m_phaseLookup.m_table.find(venueName) != m_phaseLookup.m_table.end())
   {
      if (venueInstrumentTradingStatus.Empty())
      return NONE;

      if (m_phaseLookup.m_table[venueName].find(venueInstrumentTradingStatus) != m_phaseLookup.m_table[venueName].end())
      {
         marketPhase = m_phaseLookup.m_table[venueName][venueInstrumentTradingStatus];
         TBDEBUG("Matched phases table for " << venueName << " and [" << venueInstrumentTradingStatus << "] to " << getPhaseDescription(marketPhase));
      }
      else
         TBWARNING("Looked up phases table for " << venueName << " and [" << venueInstrumentTradingStatus << "] but was not matched!");
   }

   return marketPhase;
}

void MarketPhases::getTimeBasedChanges(ContTimePairs &timePairs, const tbricks::String& venueName, const int& index)
{
   if (venueName == "Chi-X" || venueName == "BATS" || venueName == "neuro")
   {
      DateTime now = DateTime::Now();
      DateTime start = DateTime::Create(now, 9, 0, 0, 1);
      DateTime end = DateTime::Create(now, 17, 29, 58, 1);
      if(start > now)
         timePairs.push_back(ContTimePairs::value_type(start, index));
      if(end > now)
         timePairs.push_back(ContTimePairs::value_type(end, index));
   }
}

const char* MarketPhases::getPhaseDescription(const MarketPhase& marketPhase)
{
   switch (marketPhase)
   {
   case NONE: return "NONE";
   case TRADING: return "TRADING";
   case AUCTION: return "AUCTION";
   case LAST: return "LAST";
   case CLOSED: return "CLOSED";
   case HALTED: return "HALTED";
   case DISABLED: return "DISABLED";
   default: return "NONE";
   }
}


}
