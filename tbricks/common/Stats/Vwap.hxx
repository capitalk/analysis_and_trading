/* 
 * File:   VWap.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:05 PM
 */

#ifndef _STAT_VWap_HXX
#define	_STAT_VWap_HXX

#include "BaseInterface.hxx"

namespace AOStatistics
{

class VWap : public virtual BaseInterface
{
public:

   VWap(const Info& helper, const Info::Side& filterSide = Info::TOTAL) :
      BaseInterface(helper),
      _filterSide(filterSide)
   {

   }

   virtual ~VWap()
   {

   }

   virtual void start()
   {
      _vwapSum = 0;
   }

   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_filterSide == Info::TOTAL || _filterSide == side)
      {
         _vwapSum += volume * price;
      }
   }

   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0)
   {
   }

   virtual void undoTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_filterSide == Info::TOTAL || _filterSide == side)
      {
         _vwapSum -= volume * price;
      }
   }

   tbricks::Price getVWAP() const
   {
      tbricks::Volume totVolume = _refHelper.getVolume(_filterSide);
      if(totVolume > 0 && _vwapSum > 1)
         return _vwapSum / totVolume;
      else
         return tbricks::Price();
   }

protected:
   double _vwapSum;
   Info::Side _filterSide;
};

}

#endif	/* _STATVWap_HXX */

