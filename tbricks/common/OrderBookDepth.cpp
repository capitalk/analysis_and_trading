#include "OrderBookDepth.hxx"

using namespace tbricks;
using namespace common;


OrderBookEntry::OrderBookEntry(const Depth depth, const TickRule & rule)
:Depth(depth),
tickRule(rule)
{
}

// Sorts so that a std::set will have the best price first, i.e. for Side::BUY, the highest price, and the opposite for Side::SELL
bool OrderBookEntry::operator < (const OrderBookEntry & otherDepth) const
{
    if( GetDepthIdentifier() == otherDepth.GetDepthIdentifier() ) {
        return false;
    }

    // To be able to find OrderBookEntry objects in collections, we need a defined ordering between BUY & SELL objects
    if( GetSide() != otherDepth.GetSide() ) {
        return GetSide() == Side::BUY;
    }

    int price_cmp = 0;

    if (!tickRule.Empty()) {
        price_cmp = tickRule.ComparePrices(GetPrice(), otherDepth.GetPrice());
    } else {
        if (GetPrice() < otherDepth.GetPrice()) {
            price_cmp = -1;
        } else if (GetPrice() > otherDepth.GetPrice()) {
            price_cmp = 1;
        } else {
            price_cmp = 0;
        }
    }

    switch (price_cmp) {
    case -1:
        return (GetSide() == Side::SELL);
    case 0:
        if (GetSortingKey() != otherDepth.GetSortingKey()) {
            return (GetSortingKey() < otherDepth.GetSortingKey());
        }
        return (GetDepthIdentifier() < otherDepth.GetDepthIdentifier());
    case 1:
        return (GetSide() == Side::BUY);
    default:
        break;
    }

    return false;
}

void OrderBookEntry::Merge (const Depth & depth)
{
    if( depth.HasVolume() )
        SetVolume(depth.GetVolume());
    if( depth.HasPrice() )
        SetPrice(depth.GetPrice());
    if( depth.HasParticipant() )
        SetParticipant(depth.GetParticipant());
    if( depth.HasSide() )
        SetSide(depth.GetSide());
    if( depth.HasSortingKey() )
        SetSortingKey(depth.GetSortingKey());
}

bool OrderBookEntry::Empty() const
{
   return HasSide() == false && HasPrice() == false && HasVolume() == false;
}


void OrderBook::SetTickRule(const tbricks::TickRule & rule)
{
    m_tickRule = rule;
}

bool OrderBook::DepthUpdated(const tbricks::Depth & depth)
{
   if(m_cachedDepths.empty() && (m_buySide.empty() == false || m_sellSide.empty() == false))
      return false; //it is just an extract OB, created by ExtractLevelsFrom
    // First update our cached depths, and set cache_it->second to point to the complete OrderBookEntry-object
    ContainerDepths::iterator cache_it = m_cachedDepths.find(depth.GetDepthIdentifier());
    if( cache_it != m_cachedDepths.end() )
    {
        cache_it->second.Merge(depth);
    }
    else
    {
        cache_it = m_cachedDepths.insert(std::make_pair(depth.GetDepthIdentifier(), OrderBookEntry(depth, m_tickRule))).first;
//        cache_it = m_cachedDepths.find(depth.GetDepthIdentifier());
    }

    ContainerSide::iterator it;

    it = m_buySide.find(cache_it->second);
    if( it != m_buySide.end() )
    {
        m_buySide.erase(it);
        if( !depth.Deleted() )
            m_buySide.insert(cache_it->second);
    }
    else
    {
        it = m_sellSide.find(cache_it->second);
        if( it != m_sellSide.end() )
        {
            m_sellSide.erase(it);
            if( !depth.Deleted() )
                m_sellSide.insert(cache_it->second);
        }
        else if( !depth.Deleted() )
        {
            if( depth.GetSide() == Side::BUY )
                m_buySide.insert(cache_it->second);
            else
                m_sellSide.insert(cache_it->second);
        }
    }

    if( depth.Deleted() )
        m_cachedDepths.erase(cache_it);

    m_isDepthFinal = depth.IsFinal();
    
    // As of now, always returning "true"; in the future, this might change
    return true;
}

const OrderBook::ContainerSide & OrderBook::GetBuySide(void) const
{
    return m_buySide;
}

const OrderBook::ContainerSide & OrderBook::GetSellSide(void) const
{
    return m_sellSide;
}

void OrderBook::Reset(void)
{
    m_buySide.clear();
    m_sellSide.clear();
    m_cachedDepths.clear();
}

int OrderBook::GetCacheSize()
{
   return m_cachedDepths.size();
}

void OrderBook::PrintDebugInfo()
{
   TBDEBUG("Depth order book container sizes: " << m_cachedDepths.size() << ", " << m_buySide.size()
      << ", " << m_sellSide.size());
}


tbricks::Price OrderBook::GetWorstPriceForQty(const tbricks::Side& side, const tbricks::Volume &volume) const
{
   if (!m_isDepthFinal)
      return Price();

   const ContainerSide &book = side == Side::BUY ? m_buySide : m_sellSide;
   Volume volumeSum;
   for (ContainerSide::const_iterator it = book.begin(); it != book.end(); ++it)
   {
      volumeSum += it->GetVolume();
      if (volumeSum >= volume)
         return it->GetPrice();
   }

   return Price();
}

tbricks::Price OrderBook::GetAveragePriceForQty(const tbricks::Side& side, const tbricks::Volume &volume, const tbricks::Volume &skipVolume) const
{
   if (!m_isDepthFinal || volume.Empty() || volume == 0)
      return Price();

   const ContainerSide &book = side == Side::BUY ? m_buySide : m_sellSide;
   Volume skipVlm = skipVolume;
   Volume reqVlm = volume;
   double cost = 0;
   for (ContainerSide::const_iterator it = book.begin(); it != book.end(); ++it)
   {
      Volume avl = it->GetVolume();
      if(skipVlm > 0)
      {
         if(skipVlm < avl)
         {
            avl -= skipVlm;
            skipVlm = 0;
         }
         else
         {
            avl = 0;
            skipVlm -= avl;
         }
      }

      if(skipVlm == 0)
      {
         if(reqVlm < avl)
         {
            cost += reqVlm * it->GetPrice();
            //avl -= reqVlm;
            //reqVlm = 0;

            return Price(cost / volume.GetDouble());
         }
         else
         {
            cost += avl * it->GetPrice();
            //avl = 0;
            reqVlm -= avl;
         }
      }
   }

   return Price();
}

tbricks::Volume OrderBook::GetQtyForWorstPrice(const tbricks::Side& side, const tbricks::Price &price) const
{
   Volume volume;
   if (!m_isDepthFinal)
      return volume;

   const ContainerSide &book = side == Side::BUY ? m_buySide : m_sellSide;
   for (ContainerSide::const_iterator it = book.begin(); it != book.end() &&
      ((side == Side::BUY && it->GetPrice() <= price) || (side == Side::SELL && it->GetPrice() >= price)); ++it)
      volume += it->GetVolume();

   return volume;
}

const OrderBookEntry& OrderBook::GetBookRankEntry(const tbricks::Side& side, const int& rank) const
{
   if (!m_isDepthFinal)
      return emptyEntry;

   const ContainerSide &book = side == Side::BUY ? m_buySide : m_sellSide;
   ContainerSide::const_iterator it = book.begin();
   for (int entry = 1; entry < rank && it != book.end(); ++entry)
      ++it;

   return it == book.end() ? emptyEntry : *it;
}

tbricks::Integer OrderBook::GetBidRankForPrice(const tbricks::Side& side, const tbricks::Price &price) const
{
   if (!m_isDepthFinal)
      return Integer();

   Integer rank = 1;
   const ContainerSide &book = side == Side::BUY ? m_buySide : m_sellSide;
   for (ContainerSide::const_iterator it = book.begin(); it != book.end() &&
      ((side == Side::BUY && it->GetPrice() > price) || (side == Side::SELL && it->GetPrice() < price)); ++it)
      ++rank;

   return rank;
}

void OrderBook::ExtractLevelsFrom(const OrderBook& ob, const int& maxLevels)
{
   Reset();

   ContainerSide::const_iterator it = ob.GetBuySide().begin();
   ContainerSide::iterator nIt = m_buySide.begin();
   for (int entry = 0; entry < maxLevels && it != ob.GetBuySide().end(); ++entry)
   {
      nIt = m_buySide.insert(nIt, *it);
      ++it;
   }

   it = ob.GetSellSide().begin();
   nIt = m_sellSide.begin();
   for (int entry = 0; entry < maxLevels && it != ob.GetSellSide().end(); ++entry)
   {
      nIt = m_sellSide.insert(nIt, *it);
      ++it;
   }

   m_isDepthFinal = true;
}

void OrderBook::SimulateFromBBO(const tbricks::BestPrice& bp, const int& maxLevels)
{
   Reset();

   for(int side = 0; side < 2; ++side)
   {
      ContainerSide& bookSide = side ? m_sellSide : m_buySide;
      const Price& price = side ? bp.GetAskPrice() : bp.GetBidPrice();
      const Volume& volume = side ? bp.GetAskVolume() : bp.GetBidVolume();
      String id = side ? "A" : "B";
      if(price.Empty() == false && volume.Empty() == false)
         for(int dp = 0; dp < maxLevels; ++dp)
         {
            Depth entry;
            entry.SetVolume(volume.GetDouble() * (dp + 1) * (dp + 1));
            entry.SetPrice(m_tickRule.Tick(price, (side ? +1 : -1) * dp * dp));
            entry.SetSide(side ? tbricks::Side::SELL : tbricks::Side::BUY);
            id += "1";
            entry.SetDepthIdentifier(id);
            bookSide.insert(OrderBookEntry(entry, m_tickRule));
         }
   }

   m_isDepthFinal = true;
}

std::ostream &operator<<( std::ostream &strm, const OrderBook &src)
{
    int level;
    OrderBook::ContainerSide::const_iterator it;

    const OrderBook::ContainerSide & buyOrderBook(src.GetBuySide());
    it = buyOrderBook.begin();
    level = 1;
    while( it != buyOrderBook.end() )
    {
        strm << "Buy level " << level << ": " << *it;
        level++;
        it++;
        strm << "\n";
    }

    const OrderBook::ContainerSide & sellOrderBook(src.GetSellSide());
    it = sellOrderBook.begin();
    level = 1;
    while( it != sellOrderBook.end() )
    {
        strm << "Sell level " << level << ": " << *it;
        level++;
        it++;
        strm << "\n";
    }

    return strm;
}


