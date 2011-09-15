#include "ClearingParameters.hxx"
#include "tbricks_definitions.h"


using namespace tbricks;
using namespace common;

const tbricks::MIC EUREX_MIC("XEUR");
const tbricks::MIC ALLOPTIONS_MIC("ALTX");
const tbricks::MIC ALLOPTIONS2_MIC("AFET");
const tbricks::String DefaultClearingAccount("Eurex");

ClearingParameters::ClearingParameters(const tbricks::Uuid & user, const tbricks::MIC & mic, const tbricks::InstrumentIdentifier & instrument):
_user(user),
_mic(mic),
_instrument(instrument)
{  
   if(_mic == EUREX_MIC || _mic == ALLOPTIONS2_MIC || _mic == ALLOPTIONS_MIC)
   {
   	TBDEBUG("setting DefaultClearing Account "<< DefaultClearingAccount);
      _clearingTextCH = DefaultClearingAccount;
      _clearingTextNL = DefaultClearingAccount;
   }
   else
   {
	   //TBDEBUG("getting clearing from Clearing_CH and Clearing_NL");
	   std::vector<tbricks::InstrumentParameterDefinition>  parameterDefinitions;

	   parameterDefinitions.push_back(InstrumentParameterDefinition(instrument_parameters::ClearingCH()));
	   parameterDefinitions.push_back(InstrumentParameterDefinition(instrument_parameters::ClearingNL()));

	   InstrumentParameters is(instrument, parameterDefinitions);

	   is.GetParameter(instrument_parameters::ClearingCH(), _clearingTextCH);
	   is.GetParameter(instrument_parameters::ClearingNL(), _clearingTextNL);
   }
}

const tbricks::InstrumentIdentifier & ClearingParameters::GetInstrumentIdentifier()  const
{ 
	return _instrument; 
}

const tbricks::MIC & ClearingParameters::GetMIC() const
{
   return _mic;
}

const tbricks::Uuid & ClearingParameters::GetUser() const
{
   return _user;
}


tbricks::String ClearingParameters::GetClearingText() const 
{
	tbricks::String retValue;
   
   if(_mic == ALLOPTIONS_MIC || _mic == ALLOPTIONS2_MIC)
   {
   	//TBDEBUG ("clearing text returned is " << _clearingTextNL );
      return _clearingTextNL;
   }
   
	if (
		// prod Amsterdam
		_user == "2df12b4e-26d3-11df-8c15-29ffcd38ff1b" || // Abdel 
		_user == "7eea2f36-e007-11de-99c2-43a7bbf5eeeb" || // Alex
		_user == "a2b99be0-d50d-11de-9535-83937fdbcda2" || // Bjorn (TBRICKS)
		_user == "3c81ff26-1a17-11df-b855-21996720b4d2" || // Emanuele2
		_user == "50d831d6-d906-11de-b1ce-5d2abaf022ea" || // Ioannis 
		_user == "e7385906-edf4-11dd-a2a3-4fd05a6843d5" || // Ioannis (dev)
		_user == "4bd046cc-26d3-11df-bf03-978f81cd077c" || // Paul 
		_user == "31e98db0-d9dd-11de-98d4-a555518327db" || // Rogier 
		_user == "377e8892-1fca-11df-b3b4-f35c45d66911" || // Sing 
		_user == "0b3337ee-c915-11de-b030-c5023dc0299a" || // Snowdog (Sean) 
		_user == "ec83bb98-284b-11df-a39b-0b2b0c37a605" || // Spiros 
		_user == "2a8dfdfe-c915-11de-afeb-93cdd9976e21" || // Timir (prod)
		_user == "c4797614-eedc-11dd-9ba7-5d99eaf9d1fc" || // Timir (dev)
		_user == "59a3205e-c9f3-11de-a159-833baf2215d9" || // Sim (dev)
		_user == "48714d32-d8e5-11de-bd33-5b2c7bb35357" || // Wei 
		_user == "fa960dfa-cd31-11de-999e-9941a9a2d452" || // William
		
		
		// dev Amsterdam	
		_user ==  "2b7679dc-ebba-11dd-895b-c5e50b140a43" || // Sean Rolinson
		_user ==  "95195a9a-edf3-11dd-a349-6dd44d87cbca" || // William Willis
		_user ==  "e7385906-edf4-11dd-a2a3-4fd05a6843d5" || // Ioannis Mademlis
		_user ==  "c4797614-eedc-11dd-9ba7-5d99eaf9d1fc" || // Timir Karia
		_user ==  "a97b3364-fc44-11dd-8819-ddc61b4e6b97" || // Tbricks Basic User
		_user ==  "fcb7714c-0742-11de-8373-4b0f3992a60c" || // Emanuele Rocci
		_user ==  "2cfac7fe-6577-11de-816b-15e574427c53" || // Arindam Ray Mukherjee
		_user ==  "abbe903e-7dc8-11de-a4b7-77beaa6d4545" || // Cheng 		_user == Felix) Chen
		_user ==  "ae92a1e2-8c14-11de-ba45-63b1ff59ae45" || // Rogier Gerharts
		_user ==  "0541f0d4-b33d-11de-9e4f-41ba294581a9" || // Bjorn Anderson
		_user ==  "60f4dbee-b720-11de-9eae-5fea26f699f1" || // Wei Wang
		_user ==  "4804c30e-bc81-11de-93eb-d74384fd6447" || // Spiros Mourkogiannis
		_user ==  "59a3205e-c9f3-11de-a159-833baf2215d9" || // Simulation User
		_user ==  "0bad7906-d2c4-11de-86d0-cd2f51d89e63" || // Paul Tangstrom
		_user ==  "a0f0fc26-d838-11de-9c70-b3b59ddb64ff" || // Sing Kwong Cheung
		_user ==  "813da59c-e007-11de-9d8f-4db6e92c3acc"    // Alexandre Alavi
	)
	{
		retValue =_clearingTextNL;
	}
	if (
	
		// prod Zurich
		_user == "7756b950-100a-11df-8206-31ab60a6581e" ||  // Jan Arie 
		
		// dev Zurich
		_user == "5650ef70-3b01-11de-a2b5-1f2761e594d9" ||  // Jonas Haggstrom
		_user == "8b9c9dac-6fac-11de-81b2-9b1b872e345c" ||  // Jan Arie de Graaf
		_user == "4fa4d02a-bfd5-11de-8e8b-a97ede04fa0d"     // Phil Doyle
		
	)
	{
		retValue =_clearingTextCH;
	}
	
	return retValue;
}
