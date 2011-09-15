/* 
 * File:   PvpVwap.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:05 PM
 */

#ifndef _STAT_PvpVwap_HXX
#define	_STAT_PvpVwap_HXX

#include "Vwap.hxx"
#include "Pvp.hxx"

namespace AOStatistics
{

class PvpVwap : public Pvp, public VWap
{
public:

   PvpVwap(const Info& helper, const Info::Side& filterSide = Info::TOTAL) :
      BaseInterface(helper), Pvp(helper, filterSide), VWap(helper, filterSide)
   {

   }

   virtual ~PvpVwap()
   {

   }

   virtual void start()
   {
      Pvp::start();
      VWap::start();
   }

   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      Pvp::handleTrade(volume, price, time, side);
      VWap::handleTrade(volume, price, time, side);
   }

   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0)
   {
      Pvp::handleBest(side, volume, price, time);
      VWap::handleBest(side, volume, price, time);
   }

   virtual void undoTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      Pvp::undoTrade(volume, price, time, side);
      VWap::undoTrade(volume, price, time, side);
   }

   tbricks::Double getSD() const
   {
      tbricks::Volume totVolume = _refHelper.getVolume(VWap::_filterSide);
      tbricks::Price vwap = getVWAP();
      if(_mult != 0 && totVolume > 0 && vwap.Empty() == false)
      {
         vwap *= _mult;
         double variance = 0;
         //Sum[ (Vi / V) * (Pi - VWAP)^2 ]
         for(ContVolumePriceIndexByVolume::iterator it = _indexVP.get<0>().begin(); it != _indexVP.get<0>().end(); ++it)
            variance += (it->first / totVolume) * (it->second - vwap) * (it->second - vwap);

         variance /= _mult * _mult;

         if(std::abs(variance) > 1e-6)
            return std::sqrt(variance);
         else
            return tbricks::Double();
      }
      else
         return tbricks::Double();
   }

   tbricks::Double getSkew() const
   {
      tbricks::Price pvp = getPvp();
      if(pvp.Empty() == false)
      {
         tbricks::Price vwap = getVWAP();
         if(vwap.Empty() == false)
         {
            tbricks::Double sd = getSD();
            if(sd.Empty() == false)
            {
               return (vwap - pvp).GetDouble() / sd;
            }
         }
      }
      return tbricks::Double();
   }

protected:
};

}

#endif	/* _STAT_PvpVwap_HXX */

