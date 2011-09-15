/* 
 * File:   OrderBookDepth.hxx
 * Author: spiros.mourkogiannis@alloptions.nl
 *
 * Created on March 2, 2010, 1:00 PM
 */

#ifndef _ORDERBOOKDEPTH_HXX
#define	_ORDERBOOKDEPTH_HXX

#include "strategy/API.h"

#include <set>
#include <map>

#include <cmath>

namespace common
{

static const double BookLevelEqualityPrecision = 0.000001;

static const tbricks::Price DefaultPrice(0.0);
static const tbricks::Volume DefaultVolume(0.0);

/**
 * OrderBookEntry is a slightly enhanced version of the Depth class, which can be sorted
 * using the provided tick rule.
 *
 */
class OrderBookEntry :
public tbricks::Depth
{
public:
   OrderBookEntry(const tbricks::Depth depth, const tbricks::TickRule & rule);

   /**
    * Used when sorting the objects.
    *
    * Note that the sorting used is such that a normal std::set will have the best price as the first object
    *
    */
   bool operator<(const OrderBookEntry & otherDepth) const;

   /**
    * Merges a Depth object's data into the OrderBookEntry data.
    *
    * Note that the caller must first check that the merged object represents the same Depth object as
    * the OrdrBookEntry object, i.e. checking that teh depth identifiers are the same.
    *
    */
   void Merge(const tbricks::Depth & depth);

   bool Empty() const;
private:
   const tbricks::TickRule & tickRule;
};

/**
 * OrderBook class is a class which maintains a sorted orderbook consisting
 * of OrderBookEntry objects. The objects are sorted so that the first entry
 * in both the buy and sell side sets is the best price.
 */
class OrderBook
{
public:

   OrderBook() :
      m_isDepthFinal(false),
      emptyEntry(tbricks::Depth(), m_tickRule)
   {
   }

   ~OrderBook()
   {
   }

   /**
    * Setting the tick rule used when sorting.
    *
    */
   void SetTickRule(const tbricks::TickRule & rule);

   /**
    * Called when a Depth object has been updated.
    *
    *This is normally called from a HandleDepth callback.
    */
   bool DepthUpdated(const tbricks::Depth & depth);

   /**
    * Removes all depth data from the orderbook
    *
    */
   void Reset(void);

   typedef std::set<common::OrderBookEntry> ContainerSide;
   /**
    * Retrieving the sorted buy side from the orderbook
    *
    */
   const ContainerSide & GetBuySide(void) const;

   /**
    * Retrieving the sorted sell side from the orderbook
    *
    */
   const ContainerSide & GetSellSide(void) const;

   int GetCacheSize();
   void PrintDebugInfo();

   tbricks::Price GetWorstPriceForQty(const tbricks::Side& side, const tbricks::Volume &volume) const;
   tbricks::Price OrderBook::GetAveragePriceForQty(const tbricks::Side& side, const tbricks::Volume &volume, 
      const tbricks::Volume &skipVolume = tbricks::Volume(0)) const;
   tbricks::Volume GetQtyForWorstPrice(const tbricks::Side& side, const tbricks::Price &price) const;
   const OrderBookEntry& OrderBook::GetBookRankEntry(const tbricks::Side& side, const int& rank) const;
   tbricks::Integer OrderBook::GetBidRankForPrice(const tbricks::Side& side, const tbricks::Price &price) const;

   void ExtractLevelsFrom(const OrderBook& ob, const int& maxLevels = 1000);
   void SimulateFromBBO(const tbricks::BestPrice& bp, const int& maxLevels = 5);

   class BookLevel
{
public:



   //! ctor with explicit arguments

   BookLevel(const tbricks::Price & _price, const tbricks::Volume & _quantity)
      :
      price(_price),
      quantity(_quantity)
   {
   }

      BookLevel()
   :
      price(DefaultPrice),
      quantity(DefaultVolume)
   {
   }

   BookLevel(const BookLevel & bookLevel)
   :
   price(bookLevel.price),
   quantity(bookLevel.quantity)
   {
   }

   bool isVolumeValid() const
   {
      return (false == quantity.Empty());
   }

   bool isPriceValid() const
   {
      return (false == price.Empty());
   }

   bool operator==(const BookLevel &rhs) const
   {
      return (quantity == rhs.quantity) &&
         (std::fabs(price - rhs.price) < BookLevelEqualityPrecision * std::fabs(price));
   }

   bool operator !=(const BookLevel &rhs) const
   {
      return !(this->operator==(rhs));
   }
   tbricks::Price price;
   tbricks::Volume quantity;


private:
   //! ctor

   

};
   
private:
   // The sorted order books (or rather sorted OrderBookEntry objects), with the best price as the first item in each book
   ContainerSide m_buySide;
   ContainerSide m_sellSide;

   // The unsorted OrderBookEntry objects. This contains all currently non-deleted OrderBookEntry objects.
   // When HandleDepth is called, the strategy first updates the corresponding OrderBookEntry object in m_cachedDepths.
   // Then it modifies the same object in either m_buySide or m_sellSide sorted order books.
   typedef std::map<tbricks::String, common::OrderBookEntry> ContainerDepths;
   ContainerDepths m_cachedDepths;

   // The tick rule which will be used for sorting the OrderBookEntry objects
   tbricks::TickRule m_tickRule;
   bool m_isDepthFinal;
   const OrderBookEntry emptyEntry;
};


}

std::ostream &operator<<( std::ostream &strm, const common::OrderBook &src);

#endif	/* _ORDERBOOKDEPTH_HXX */

