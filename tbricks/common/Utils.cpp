
#include "strategy/API.h"


tbricks::Double getValueFactor(const tbricks::InstrumentIdentifier &id)
{
    tbricks::Instrument theInstrument(id);

    if (false == theInstrument.HasXML())
    {
        TBDEBUG("!!WARNING. NO XML IS ASSOCIATED WITH INSTRUMENT. SETTING VALUE FACTOR TO 1 for instrument id: " << id);
        return 1;
        //throw std::runtime_error("NO XML IS ASSOCIATED WITH INSTRUMENT. CANNOT SET THE VALUE FACTOR. Aborting...");
    }

    tbricks::XMLDocument XML = theInstrument.GetXML();
    tbricks::XMLNode node = XML.GetDocumentElement();
    
    tbricks::String stringValue;
	tbricks::Double valueFactor;
    if (false == node.GetValue("/FIXML/SecDef/Instrmt/@Fctr", valueFactor) || valueFactor.Empty())
    {
        TBWARNING("!!!! > WARNING. No XML for instrument - setting @fctr to 1 for instrument id: " << id );
        return 1;
        //throw std::runtime_error("CANNOT ACCESS /FIXML/SecDef/Instrmt/@Fctr. CANNOT SET THE VALUE FACTOR. Aborting...");
    }
    TBDEBUG("Value factor " << id << ":"  << valueFactor);

    //return boost::lexical_cast<double>(stringValue.GetCString());
	return valueFactor;
}



