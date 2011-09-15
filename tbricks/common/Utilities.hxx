/* 
 * File:   MarketPhases.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on May 21, 2010, 5:47 PM
 */

#ifndef _UTILITIES_HXX
#define	_UTILITIES_HXX

#include "strategy/API.h"
#include <vector>

namespace common
{

tbricks::Venue getVenueFromShortName(const tbricks::String& shortName);

bool getCompositeInstrumentIvidsFromXML(const tbricks::Instrument & composite,
   std::vector<tbricks::InstrumentVenueIdentification>& ivids, bool bLog = false);
bool getCompositeInstrumentIvidsAndRatiosFromXML(const tbricks::Instrument & composite,
   std::vector<tbricks::InstrumentVenueIdentification>& ivids,std::vector<double>& ratios, bool bLog = false);
struct sortCompositeOnRatio
{
public:
   bool operator()(const tbricks::InstrumentComponent& lhs, const tbricks::InstrumentComponent& rhs)
   {
       return (lhs.GetRatio() < rhs.GetRatio());
   }
};

double getImportantDigits(const double& value, const int n);

double getValueFactor(const tbricks::InstrumentIdentifier &id);
}

#endif	/* _UTILITIES_HXX */

