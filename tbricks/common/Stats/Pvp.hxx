/* 
 * File:   Pvp.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:05 PM
 */

#ifndef _STAT_Pvp_HXX
#define	_STAT_Pvp_HXX

#include "BaseInterface.hxx"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include "TemplateHelpers.hxx"

namespace AOStatistics
{

class Pvp : public virtual BaseInterface
{
public:

   Pvp(const Info& helper, const Info::Side& filterSide = Info::TOTAL) :
      BaseInterface(helper),
      _filterSide(filterSide)
   {

   }

   virtual ~Pvp()
   {

   }

   virtual void start()
   {
      _indexVP.clear();
      _mult = 0;
   }

   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_filterSide == Info::TOTAL || _filterSide == side)
      {
         if(_mult == 0 && std::abs(price) > 1e-3)
            _mult = std::pow(10.0, 5 - int(std::log10(std::fabs(price))) - 1); //keep n = 5 important digits

         PriceHashType keyPrice = PriceHashType(price * _mult + 0.5);
         ContVolumePriceIndex::iterator it = _indexVP.insert(PairVolumePrice(0, keyPrice)).first;  //if already exists, we will get an iterator to the existing item
         _indexVP.modify(it,
            common::IncrementBy <PairVolumePrice::first_type, PairVolumePrice, PairVolumePriceExtractor1st > (volume));
         //TBSTATUS("Traded " << volume << " @ " << price << " => totVol = " << _indexVP.get<1>().find(keyPrice)->first << ", PVP = " << getPvp());
      }
   }

   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0)
   {
   }

   virtual void undoTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE)
   {
      if (_filterSide == Info::TOTAL || _filterSide == side)
      {
         PriceHashType keyPrice = PriceHashType(price * _mult + 0.5);
         ContVolumePriceIndexByPrice::iterator it = _indexVP.get<1>().find(keyPrice);
         if(it != _indexVP.get<1>().end())
            _indexVP.modify(_indexVP.find(*it),
               common::IncrementBy <PairVolumePrice::first_type, PairVolumePrice, PairVolumePriceExtractor1st > (-volume));
      }
   }

   tbricks::Price getPvp() const
   {
      if(_indexVP.empty() || _mult == 0)
         return tbricks::Price();
      else
         return tbricks::Price(_indexVP.get<0>().begin()->second / _mult);
   }

protected:
   Info::Side _filterSide;
   typedef unsigned short PriceHashType;
   typedef std::pair<tbricks::Volume, PriceHashType> PairVolumePrice;
   typedef boost::multi_index::member< PairVolumePrice, PairVolumePrice::first_type, &PairVolumePrice::first > PairVolumePriceExtractor1st;
   typedef boost::multi_index::member< PairVolumePrice, PairVolumePrice::second_type, &PairVolumePrice::second > PairVolumePriceExtractor2nd;
   typedef boost::multi_index_container< PairVolumePrice,
                                         boost::multi_index::indexed_by <
                                                                           boost::multi_index::ordered_unique< boost::multi_index::identity<PairVolumePrice>, std::greater<PairVolumePrice>  >, // sort by inverse PairVolumePrice::operator<
                                                                           boost::multi_index::hashed_unique< PairVolumePriceExtractor2nd >
                                                                       >
                                       > ContVolumePriceIndex;
   typedef ContVolumePriceIndex::nth_index<0>::type ContVolumePriceIndexByVolume;
   typedef ContVolumePriceIndex::nth_index<1>::type ContVolumePriceIndexByPrice;

   ContVolumePriceIndex _indexVP;
   double _mult;
};

}

#endif	/* _STATPvp_HXX */

