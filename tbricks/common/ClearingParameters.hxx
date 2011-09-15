/* 
 * File:   OrderBookDepth.hxx
 *
 * Created on March 8, 2010, 1:00 PM
 */

#ifndef _CLEARINGPARAMETER_HXX
#define	_CLEARINGPARAMETER_HXX

#include "strategy/API.h"

#include <set>
#include <map>

namespace common
{

/**
 * ClearingParameter gets information set in the InstrumentParameter in order to set the clearing text of the instrument correctly. 
 * ClearingParameter must be set using a combination of User and Instrument
 * 1) Needs user in the ctor to determine the location of the strategy (Swiss trades must be routed differently than Dutch EVEN to the same exchange)
 * 2) Instrument is the instrument that we will be placing the order for 
 *
 */
class ClearingParameters 
{
public:
	ClearingParameters(const tbricks::Uuid & user, const tbricks::MIC & mic, const tbricks::InstrumentIdentifier & instrument);
   
	const tbricks::Uuid & GetUser() const;
   
   const tbricks::MIC & GetMIC() const;

	const tbricks::InstrumentIdentifier & GetInstrumentIdentifier() const;

	tbricks::String GetClearingText() const;


private:
	tbricks::InstrumentIdentifier _instrument; 
	tbricks::MIC _mic; 
	tbricks::Uuid _user;
	tbricks::Venue _venue;
	tbricks::String _clearingTextCH;
	tbricks::String _clearingTextNL;

};

}

#endif // _CLEARINGPARAMETER_HXX
