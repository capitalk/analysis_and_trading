/* 
 * File:   Container.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on June 22, 2010, 12:05 PM
 */

#ifndef _STAT_CONTAINER_HXX
#define	_STAT_CONTAINER_HXX

#include "BaseInterface.hxx"
#include <strategy/API.h>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace AOStatistics
{

class Container : public BaseInterface
{
public:
   Container(const double& readyTime = 0);
   virtual ~Container();

   void setTickRule(const tbricks::TickRule& rule);
   virtual void start();
   virtual void handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0, Info::Side side = Info::NONE);
   virtual void handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time = 0);
   bool isReady() const;

   const Info& getInfo() const;
   size_t add(BaseInterface* pComponent);
   size_t add(const boost::shared_ptr<BaseInterface>& component);
   const boost::shared_ptr<BaseInterface>& get(const size_t& index) const;
   const boost::shared_ptr<BaseInterface>& operator [] (const size_t& index) const;
   void clear();  
   
protected:
   Info _helper;
   typedef std::vector< boost::shared_ptr<BaseInterface> > ContStats;
   ContStats _stats;
   const boost::shared_ptr<BaseInterface> _emptySharedPtr;
   const double _readyTime;
   bool _ready;
   tbricks::Double _time1st;
};

}

#endif	/* _STATCONTAINER_HXX */

