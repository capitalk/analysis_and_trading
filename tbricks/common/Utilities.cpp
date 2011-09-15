#include "Utilities.hxx"

#include <cmath>
#include <boost/lexical_cast.hpp>

namespace common
{

using namespace tbricks;

Venue getVenueFromShortName(const String& shortName)
{
   std::vector<Venue> venues;
   VenueManager::GetAllVenues(venues);

   for (size_t i = 0; i < venues.size(); ++i)
   {
      if (venues[i].GetShortName() == shortName)
         return venues[i];
   }

   return Venue();
}


//Helper class for finding an instrument in a container of InstrumentComponent, using it's short name

class FindInstrumentHelper
{
public:

   FindInstrumentHelper(const tbricks::String& shortName) : m_shortName(shortName)
   {
   }

   bool operator() (const tbricks::InstrumentComponent& component) const
   {
      return tbricks::Instrument(component.GetInstrumentIdentifier()).GetShortName() == m_shortName;
   }

private:
   const tbricks::String& m_shortName;
};

/* Example format: (check _CIP1 in dev)
<Custom>
   <Instrument_0 ShortName="ALVd" Venue="Xetra" MIC="XETR"/>
   <Instrument_1 ShortName="ALVd" Venue="Chi-X" MIC="CHIX"/>
   <Instrument_2 ShortName="DBKd" Venue="Xetra" MIC="XETR"/>
   <Instrument_3 ShortName="DBKd" Venue="Chi-X" MIC="CHIX"/>
   <Instrument_4 ShortName="SIEd" Venue="Xetra" MIC="XETR"/>
   <Instrument_5 ShortName="SIEd" Venue="Chi-X" MIC="CHIX"/>
</Custom>
 */
bool getCompositeInstrumentIvidsFromXML(const tbricks::Instrument & composite,
   std::vector<tbricks::InstrumentVenueIdentification>& ivids, bool bLog)
{
   ivids.clear();
   if (!composite.HasXML())
   {
      if (bLog)
         TBERROR("No XML associated with instrument " << composite.GetShortName());
      return false;
   }

   TBDEBUG("Getting XML for " << composite.GetShortName());
   std::vector<tbricks::InstrumentComponent> components;
   composite.GetComponents(components);
   int index = 0;
   do
   {
      String node = "/Custom/Instrument_";
      node += boost::lexical_cast<std::string > (index).c_str();

      bool bSuc = true;
      String instrName;
      String node0(node);
      node0 += "/@ShortName";
      if (false == composite.GetXML().GetDocumentElement().GetValue(node0, instrName) || instrName.Empty())
      {
         TBDEBUG("No node found in XML of composite for index" << index);
         break;
      }

      std::vector<tbricks::InstrumentComponent>::iterator itComp =
         std::find_if(components.begin(), components.end(), FindInstrumentHelper(instrName));
      if (itComp == components.end())
      {
         if (bLog)
            TBERROR("Instrument short name " << instrName << " not found in the components of composite instrument " << composite.GetShortName());
         ivids.clear();
         return false;
      }

      String venueName;
      String node1(node);
      node1 += "/@Venue";
      Venue venue;
      if (false == composite.GetXML().GetDocumentElement().GetValue(node1, venueName) ||
         venueName.Empty() || (venue = common::getVenueFromShortName(venueName)).Empty())
      {
         if (bLog)
            TBERROR("No/invalid Venue XML for index " << index << " of composite instrument " << composite.GetShortName() <<
            " in node " << node1 << ": " << venueName);
         ivids.clear();
         return false;
      }

      String micName;
      String node2(node);
      node2 += "/@MIC";
      MIC mic;
      if (false == composite.GetXML().GetDocumentElement().GetValue(node2, micName) ||
         micName.Empty() || (mic = MIC(micName.GetCString())).Empty())
      {
         if (bLog)
            TBERROR("No/invalid MIC XML for index " << index << " of composite instrument " << composite.GetShortName() <<
            " in node " << node2 << ": " << micName);
         ivids.clear();
         return false;
      }

      ivids.push_back(tbricks::InstrumentVenueIdentification(itComp->GetInstrumentIdentifier(), venue, mic));
      TBDEBUG("Resolved tradeable of composite for index " << index << " is: " << ivids.back());
      ++index;
   }
   while (true);

   return ivids.size() > 0;
}

bool getCompositeInstrumentIvidsAndRatiosFromXML(const tbricks::Instrument & composite,
   std::vector<tbricks::InstrumentVenueIdentification>& ivids,std::vector<double>& ratios, bool bLog)
{

   ivids.clear();
   ratios.clear();

   if (!composite.HasXML())
   {
      if (bLog)
         TBERROR("No XML associated with instrument " << composite.GetShortName());
      return false;
   }

   TBDEBUG("Getting XML for " << composite.GetShortName());
   std::vector<tbricks::InstrumentComponent> components;
   composite.GetComponents(components);
   int index = 0;
   do
   {
      String node = "/Custom/Instrument_";
      node += boost::lexical_cast<std::string > (index).c_str();

      bool bSuc = true;
      String instrName;
      String node0(node);
      node0 += "/@ShortName";
      if (false == composite.GetXML().GetDocumentElement().GetValue(node0, instrName) || instrName.Empty())
      {
         TBDEBUG("No node found in XML of composite for index" << index);
         break;
      }

      std::vector<tbricks::InstrumentComponent>::iterator itComp =
         std::find_if(components.begin(), components.end(), FindInstrumentHelper(instrName));
      if (itComp == components.end())
      {
         if (bLog)
            TBERROR("Instrument short name " << instrName << " not found in the components of composite instrument " << composite.GetShortName());
         ivids.clear();
         return false;
      }

      double ratio = itComp->GetRatio().GetDouble();

      String venueName;
      String node1(node);
      node1 += "/@Venue";
      Venue venue;
      if (false == composite.GetXML().GetDocumentElement().GetValue(node1, venueName) ||
         venueName.Empty() || (venue = common::getVenueFromShortName(venueName)).Empty())
      {
         if (bLog)
            TBERROR("No/invalid Venue XML for index " << index << " of composite instrument " << composite.GetShortName() <<
            " in node " << node1 << ": " << venueName);
         ivids.clear();
         return false;
      }

      String micName;
      String node2(node);
      node2 += "/@MIC";
      MIC mic;
      if (false == composite.GetXML().GetDocumentElement().GetValue(node2, micName) ||
         micName.Empty() || (mic = MIC(micName.GetCString())).Empty())
      {
         if (bLog)
            TBERROR("No/invalid MIC XML for index " << index << " of composite instrument " << composite.GetShortName() <<
            " in node " << node2 << ": " << micName);
         ivids.clear();
         return false;
      }

      ivids.push_back(tbricks::InstrumentVenueIdentification(itComp->GetInstrumentIdentifier(), venue, mic));
      ratios.push_back(ratio);
      TBDEBUG("Resolved tradeable of composite for index " << index << " is: " << ivids.back());
      ++index;
   }
   while (true);

   return ivids.size() > 0;
}

double getImportantDigits(const double& value, const int n)
{
   if (std::abs(value) < 1e-8)
      return value;
   double decFactor = std::pow(10.0, n - int(std::log10(std::fabs(value))) - 1); //always show n important digits
   return int(value * decFactor + (value < 0 ? -0.5 : +0.5)) / decFactor;
}

double getValueFactor(const tbricks::InstrumentIdentifier &id)
{
   tbricks::Instrument theInstrument(id);

   if (false == theInstrument.HasXML())
   {
      TBDEBUG("!!WARNING. NO XML IS ASSOCIATED WITH INSTRUMENT. SETTING VALUE FACTOR TO 1");
      return 1;
      //throw std::runtime_error("NO XML IS ASSOCIATED WITH INSTRUMENT. CANNOT SET THE VALUE FACTOR. Aborting...");
   }

   tbricks::XMLDocument XML = theInstrument.GetXML();
   tbricks::XMLNode node = XML.GetDocumentElement();

   tbricks::String stringValue;
   if (false == node.GetValue("/FIXML/SecDef/Instrmt/@Fctr", stringValue) || stringValue.Empty())
   {
      TBDEBUG("!!WARNING. NO XML IS ASSOCIATED WITH INSTRUMENT. SETTING VALUE FACTOR TO 1");
      return 1;
      throw std::runtime_error("CANNOT ACCESS /FIXML/SecDef/Instrmt/@Fctr. CANNOT SET THE VALUE FACTOR. Aborting...");
   }
   TBDEBUG("SETTING VALUE FACTOR TO " << stringValue);

   return boost::lexical_cast<double>(stringValue.GetCString());
}


}
