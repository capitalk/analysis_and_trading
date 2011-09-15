#include "Container.hxx"

namespace AOStatistics
{

Container::Container(const double& readyTime) : BaseInterface(_helper),
   _readyTime(readyTime)
{

}

Container::~Container()
{

}

void Container::setTickRule(const tbricks::TickRule& rule)
{
   _helper._rule = rule;
}

void Container::start()
{
   _ready = false;
   _time1st.Clear();
   _helper.reset();
   for (ContStats::iterator it = _stats.begin(); it != _stats.end(); ++it)
   {
      (*it)->start();
   }
}

void Container::handleTrade(const tbricks::Volume& volume, const tbricks::Price& price, const double& time, Info::Side side)
{
   if(_time1st.Empty())
      _time1st = time;
   if((time - _time1st) >= _readyTime)
      _ready = true;

   if(side >= Info::NONE)
      side = _helper.getTradeSide(price);

   ++_helper._countTrades[side];
   _helper._countVolume[side] += volume;
   _helper._lastPrice[side] = price;
   _helper._lastPrice[Info::NONE] = price;

   for (ContStats::iterator it = _stats.begin(); it != _stats.end(); ++it)
   {
      (*it)->handleTrade(volume, price, time, side);
   }
}
void Container::handleBest(const Info::Side& side, const tbricks::Volume& volume, const tbricks::Price& price, const double& time)
{
   _helper._bestPrice[side] = price;

   for (ContStats::iterator it = _stats.begin(); it != _stats.end(); ++it)
   {
      (*it)->handleBest(side, volume, price, time);
   }
}

bool Container::isReady() const
{
   return _ready;
}

const Info& Container::getInfo() const
{
   return _helper;
}

size_t Container::add(BaseInterface* pComponent)
{
   return add(boost::shared_ptr<BaseInterface>(pComponent));
}

size_t Container::add(const boost::shared_ptr<BaseInterface>& component)
{
   size_t index = _stats.size();
   _stats.push_back(component);
   
   return index;
}

const boost::shared_ptr<BaseInterface>& Container::get(const size_t& index) const
{
   if(index < _stats.size())
      return _stats[index];
   else
      return _emptySharedPtr;
}

const boost::shared_ptr<BaseInterface>& Container::operator [] (const size_t& index) const
{
   if(index < _stats.size())
      return _stats[index];
   else
      return _emptySharedPtr;
}

void Container::clear()
{
   _stats.clear(); //@Spiros why only _stats.clear and not _helper.reset()? Is this meant to be only "clearStats"?
   //@Ioannis: the _helper gets reset every time we start() the container (e.g. for a new day) - this clear is only used for clearing the stat functions contained by this object.
}

}
