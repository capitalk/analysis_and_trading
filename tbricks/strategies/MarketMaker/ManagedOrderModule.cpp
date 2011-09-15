/*
 *  ManagedOrderModule.cpp
 *  Strategies
 *
 *  Created by Konstantin Romanov on 7/8/09.
 *  Copyright 2009-2011 Tbricks. All rights reserved.
 *
 */

#include "ManagedOrderModule.h"

#define MODULE_DEBUG(a) TBDEBUG(m_name << ": " << a)
#define MODULE_STATUS(a) TBSTATUS(m_name << ": " << a)

#define MODULE_EXTRA_DATA_KEY "Module"

ManagedOrderModule::ManagedOrderModule(const Strategy& parentStrategy, 
                                       SharedOrderManager& orderManager,
                                       const InstrumentVenueIdentification& ivid, 
                                       const PortfolioIdentifier& portfolio,
                                       ManagedOrderModule::IModuleHandler& handler,
                                       const String& name)
:m_strategy(parentStrategy),
m_orderManager(orderManager),
m_handler(handler),
m_name(name),
m_ivid(ivid),
m_portfolioId(portfolio),
m_tradingInfo(ivid),
m_instrument(ivid.GetInstrumentIdentifier()),
m_state(ManagedOrderState_Inactive),
m_totalFilledVolume(0),
m_currentOrderFilledVolume(0),
m_hidden(false)
{
  MODULE_DEBUG("ManagedOrderModule::ManagedOrderModule");
  m_orderManager.AddManager(*this);
}

ManagedOrderModule::~ManagedOrderModule()
{
  MODULE_DEBUG("ManagedOrderModule::~ManagedOrderModule");
  m_orderManager.RemoveManager(*this);
}

void ManagedOrderModule::SetHidden(bool hidden)
{
  if (m_hidden != hidden)
  {
    m_hidden = hidden;
  }
}

void ManagedOrderModule::SetMinimumVolume(const Volume& volume)
{
  if (m_minimumVolume != volume)
  {
    m_minimumVolume = volume;
  }
}


void ManagedOrderModule::CreateActivationConditions(const Side& side, const Price& price, Order::ActivationConditions& conditions) const
{
  if (side == Side::BUY)
  {
    if (!m_minimumVolume.Empty())
    {
      conditions.Add(m_ivid, BestPriceByAskFilter(price, Filter::LE) && BestPriceByAskVolumeFilter(m_minimumVolume, Filter::GE));
    }
    else
    {
      conditions.Add(m_ivid, BestPriceByAskFilter(price, Filter::LE));
    }
  }
  else
  {
    if (!m_minimumVolume.Empty())
    {
      conditions.Add(m_ivid, BestPriceByBidFilter(price, Filter::GE) && BestPriceByBidVolumeFilter(m_minimumVolume, Filter::GE));
    }
    else
    {
      conditions.Add(m_ivid, BestPriceByBidFilter(price, Filter::GE));
    }
  }
}



void ManagedOrderModule::ModifyPriceAndActiveVolume(const Price& price, const Volume& volume, bool forceUpdateVolume)
{
  MODULE_DEBUG("ManagedOrderModule::ModifyPriceAndActiveVolumee(" << price << ", " << volume << ")");

  if (GetState() == ManagedOrderState_Inactive)
  {
    TBERROR("Internal error: attempt to modify price and active volume in inactive state.");
    return;
  }
  
  if (price.Empty() || (volume.GetType() != Volume::TOTAL))
  {
    TBERROR("ManagedOrderModule::ModifyPriceAndActiveVolume() price and volume must be set.");
    return;
  }

  bool shouldModifyPrice = true;
  bool shouldModifyVolume = true;

  if (!m_pendingPrice.Empty())
  {
    // we have pending price modification - let's check if we should change again
    if (m_tradingInfo.GetTickRule().ComparePrices(m_pendingPrice, price) == 0)
    {
      MODULE_DEBUG("Request to modify price to " << price << " which is already pending, ignored.");
      shouldModifyPrice = false;
    }
  }
  else
  {
    // no pending price - let's check with the order's actual price
    Price currentPrice;
    m_order.GetPrice(currentPrice);
    
    if (m_tradingInfo.GetTickRule().ComparePrices(currentPrice, price) == 0)
    {
      MODULE_DEBUG("Request to modify price to " << price << " which is already on market, ignored.");
      shouldModifyPrice = false;
    }
  }
  
  if (!forceUpdateVolume)
  {
    if (!m_pendingVolume.Empty())
    {
      // we have pending volume modification - let's check if we should change again
      if (m_tradingInfo.CompareVolumes(m_pendingVolume, volume) == 0)
      {
        MODULE_DEBUG("Request to modify volume to " << volume << " which is already pending, ignored.");
        shouldModifyVolume = false;
      }
    }
    else
    {
      // no pending volume - let's check with the order's actual volume
      Volume currentVolume;
      m_order.GetActiveVolume(currentVolume);
      
      if (m_tradingInfo.CompareVolumes(currentVolume, volume) == 0)
      {
        MODULE_DEBUG("Request to modify volume to " << volume << " which is already on market, ignored.");
        shouldModifyVolume = false;
      }
    }
  }
  
  if (!shouldModifyPrice && !shouldModifyVolume)
  {
    MODULE_DEBUG("Neither price nor volume should be changed.");
    return;
  }

  m_pendingPrice = price;
  
  Order::Modifier modifier;

  if (shouldModifyPrice)
  {
    modifier.SetPrice(price);
    m_pendingPrice = price;
  }
  if (shouldModifyVolume)
  {
    modifier.SetActiveVolume(volume);
    m_pendingVolume = volume;
  }

  if (m_hidden)
  {
    CreateActivationConditions(m_pendingSide, m_pendingPrice, modifier.GetActivationConditions());
  }
  
  if ((shouldModifyPrice && !CanModifyPrice()) || (shouldModifyVolume && !CanModifyVolume()))
  {
    SetState(ManagedOrderState_PendingModify);

    Identifier requestId = m_orderManager.SendDeleteRequest(*this, m_order.GetIdentifier());
    MODULE_DEBUG("Send SYNTHETIC modify (DELETE) request for order: " << m_order.GetIdentifier() << ", requestId: " << requestId);
  }
  else
  {
    Identifier requestId = m_orderManager.SendModifyRequest(*this, m_order.GetIdentifier(), modifier);
    MODULE_DEBUG("Send modify request for order: " << m_order.GetIdentifier() << ", requestId: " << requestId);
  }
}

void ManagedOrderModule::ModifyPrice(const Price& price)
{
  MODULE_DEBUG("ManagedOrderModule::ModifyPrice(" << price << ")");
  
  if (GetState() == ManagedOrderState_Inactive)
  {
    TBERROR("Internal error: attempt to modify price in inactive state.");
    return;
  }
  
  if (!m_pendingPrice.Empty())
  {
    // we have pending price modification - let's check if we should change again
    if (m_tradingInfo.GetTickRule().ComparePrices(m_pendingPrice, price) == 0)
    {
      MODULE_DEBUG("Request to modify price to " << price << " which is already pending, ignored.");
      return;
    }
  }
  else
  {
    // no pending price - let's check with the order's actual price
    Price currentPrice;
    m_order.GetPrice(currentPrice);
    
    if (m_tradingInfo.GetTickRule().ComparePrices(currentPrice, price) == 0)
    {
      MODULE_DEBUG("Request to modify price to " << price << " which is already on market, ignored.");
      return;      
    }
  }
  
  m_pendingPrice = price;
  
  Order::Modifier modifier;

  modifier.SetPrice(price);

  if (m_hidden)
  {
    CreateActivationConditions(m_pendingSide, price, modifier.GetActivationConditions());
  }

  if (!CanModifyPrice())
  {
    Identifier requestId = m_orderManager.SendModifyRequest(*this, m_order.GetIdentifier(), modifier);
    MODULE_DEBUG("Send modify request for price " << price << " order: " << m_order.GetIdentifier() << ", requestId: " << requestId);
  }
  else
  {
    SetState(ManagedOrderState_PendingModify);
    Identifier requestId = m_orderManager.SendDeleteRequest(*this, m_order.GetIdentifier());
    MODULE_DEBUG("Send SYNTHETIC modify (DELETE) request for price " << price << " order: " << m_order.GetIdentifier() << ", requestId: " << requestId);
  }
}

void ManagedOrderModule::IncreaseActiveVolume(const Volume& volume)
{
  MODULE_DEBUG("ManagedOrderModule::IncreateActiveVolume(" << volume << ")");
}

void ManagedOrderModule::DecreaseActiveVolume(const Volume& volume)
{
  MODULE_DEBUG("ManagedOrderModule::DecreaseActiveVolume(" << volume << ")");
}

void ManagedOrderModule::ModifyActiveVolume(const Volume& volume, bool forceUpdateVolume)
{
  MODULE_DEBUG("ManagedOrderModule::ModifyActiveVolume(" << volume << ")");
  
  if (GetState() == ManagedOrderState_Inactive)
  {
    TBERROR("Internal error: attempt to modify volume in inactive state.");
    return;
  }
  
  if (!forceUpdateVolume)
  {
    if (!m_pendingVolume.Empty())
    {
      // we have pending volume modification - let's check if we should change again
      if (m_tradingInfo.CompareVolumes(m_pendingVolume, volume) == 0)
      {
        MODULE_DEBUG("Request to modify volume to " << volume << " which is already pending, ignored.");
        return;
      }
    }
    else
    {
      // no pending volume - let's check with the order's actual volume
      Volume currentVolume;
      m_order.GetActiveVolume(currentVolume);
      
      if (m_tradingInfo.CompareVolumes(currentVolume, volume) == 0)
      {
        MODULE_DEBUG("Request to modify volume to " << volume << " which is already on market, ignored.");
        return;      
      }
    }
  }

  m_pendingVolume = volume;
  
  Order::Modifier modifier;
  modifier.SetActiveVolume(volume);

  if (!CanModifyVolume())
  {
    Identifier requestId = m_orderManager.SendModifyRequest(*this, m_order.GetIdentifier(), modifier);  
    MODULE_DEBUG("Send modify request for volume " << volume << " order: " << m_order.GetIdentifier() << ", requestId: " << requestId);
  }
  else
  {
    SetState(ManagedOrderState_PendingModify);
    Identifier requestId = m_orderManager.SendDeleteRequest(*this, m_order.GetIdentifier());
    MODULE_DEBUG("Send SYNTHETIC modify (DELETE) request for volume " << volume << " order: " << m_order.GetIdentifier() << ", requestId: " << requestId);
  }

  
}

void ManagedOrderModule::Send(const Side& side, const Volume& volume, const Price& price)
{
  MODULE_DEBUG("ManagedOrderModule::Activate()");
  
  if (GetState() != ManagedOrderState_Inactive)
  {
    TBERROR("Internal error: attempt to call Send() on already existing order.");
    return;
  }
  
  Order::Options options;
  options.SetInstrumentVenueIdentification(m_ivid);
  options.SetPrice(price);
  options.SetActiveVolume(volume);
  options.SetSide(side);
  options.SetPortfolioIdentifier(m_portfolioId);
  options.GetExtraData().SetValue(MODULE_EXTRA_DATA_KEY, m_name);

  m_pendingVolume = volume;
  m_pendingPrice = price;
  m_pendingSide = side;

  m_priceInCreateRequest = price;

  m_currentOrderFilledVolume = 0;

  if (m_hidden)
  {
    CreateActivationConditions(side, price, options.GetActivationConditions());

    if (VenueSupportsFAK())
    {
      options.SetValidity(Validity::VALID_IMMEDIATE);
    }
  }

  if (options.GetValidity().Empty())
  {
    const ValidityTypeSet& validities = m_ivid.GetVenue().GetTradingCapabilities().GetSupportedValidityTypes();
    if (validities.Has(Validity::VALID_TODAY))
    {
      options.SetValidity(Validity::VALID_TODAY);
    }
    else if (validities.Has(Validity::VALID_INFINITE))
    {
      options.SetValidity(Validity::VALID_INFINITE);
    }
    else
    {
      // will be rejected by SF, don't support strange validities for now
    }
  }

  OrderManager::OrderCreateRequestResult result = m_orderManager.SendCreateRequest(*this, options);
  m_order = Order(result.GetOrderIdentifier());
  SetState(ManagedOrderState_PendingCreate);
  
  MODULE_DEBUG("Sent order create request for order :" << result.GetOrderIdentifier() << ", request id: " << result.GetRequestIdentifier());
}

void ManagedOrderModule::Cancel()
{
  MODULE_DEBUG("ManagedOrderModule::Deactivate()");

  if (GetState() == ManagedOrderState_Inactive)
  {
    MODULE_DEBUG("Attempt to cancel an already inactive order, ignored.");
    return;
  }
  
  Identifier requestId = m_orderManager.SendDeleteRequest(*this, m_order.GetIdentifier());
  SetState(ManagedOrderState_PendingDelete);
  
  MODULE_DEBUG("Deleting order " << m_order.GetIdentifier() << ", requestId: " << requestId);
}


Volume ManagedOrderModule::GetActiveVolume() const
{
  Volume volume;
  m_order.GetActiveVolume(volume);
  return volume;
}

Price ManagedOrderModule::GetPrice() const
{
  if (GetState() == ManagedOrderState_PendingCreate)
  {
    // When doing pending create, let's consider price being added as the real one
    return m_priceInCreateRequest;
  }
  else
  {
    Price price;
    m_order.GetPrice(price);
    return price;
  }
}

Price ManagedOrderModule::GetPendingPrice() const
{
  return m_pendingPrice;
}


void ManagedOrderModule::HandleRecoveryCompleted()
{
  MODULE_DEBUG("ManagedOrderModule::HandleRecoveryCompleted()");
}

bool ManagedOrderModule::HandleOrderRecovery(const Order::Update& update)
{
  MODULE_DEBUG("ManagedOrderModule::HandleOrderRecovery()");
  
  ExtraData extraData;
  if (update.HasExtraData() && update.GetExtraData(extraData))
  {
    String moduleName;
    if (extraData.GetValue(MODULE_EXTRA_DATA_KEY, moduleName) && (moduleName == m_name))
    {
      Volume filledVolume;
      if (update.HasFilledVolume())
      {
        update.GetFilledVolume(filledVolume);
        m_totalFilledVolume += filledVolume;
      }

      Boolean deleted;
      if (update.GetDeleted(deleted) && !deleted)
      {
        m_order = Order();
        m_order.Merge(update);
        SetState(ManagedOrderState_Active);
        m_order.GetActiveVolume(m_pendingVolume);
        m_order.GetPrice(m_pendingPrice);
        m_order.GetSide(m_pendingSide);

        TBSTATUS("Recovered active order " << update);
      }

      m_handler.ManagedOrderModule_HandleOrderUpdate(*this, update);
      
      MODULE_DEBUG("Order " << update << " recovered.");

      return true;
    }
  }
  
  return false;
}


bool ManagedOrderModule::VenueSupportsFAK() const
{
  return m_ivid.GetVenue().GetTradingCapabilities().GetSupportedValidityTypes().Has(Validity::VALID_IMMEDIATE);
}

void ManagedOrderModule::HandleOrderUpdate(const Order::Update& update)
{
  MODULE_DEBUG("ManagedOrderModule::HandleOrderUpdate(): " << update);
  
  if (update.GetIdentifier() != m_order.GetIdentifier())
  {
    TBDEBUG("Received already obsolete order. Own: " << m_order << " update: " << update);    
    // already obsolete
    return;
  }
  
  if (update.HasTradingLimitState())
  {
    TradingLimitState tradingLimitState;
    update.GetTradingLimitState(tradingLimitState);

    switch (tradingLimitState.Get())
    {
    case TradingLimitState::OK:
      // everything's fine
      break;

    case TradingLimitState::NO_RULES:
      {
        Alert alert(String("Order failed: no trading limits configured for ") + m_instrument.GetShortName());
        alert.AddRelatedInstrument(m_instrument.GetIdentifier());
        alert.Send();
      }

      TBWARNING("Order failed: no trading limits configured for " << m_instrument.GetShortName());
      break;

    case TradingLimitState::SOFT_BREACH:
      TBNOTICE("Soft limit breach in " << m_instrument.GetShortName());
      break;

    case TradingLimitState::HARD_BREACH:
      {
        Alert alert(String("Hard limit breach in ") + m_instrument.GetShortName());
        alert.AddRelatedInstrument(m_instrument.GetIdentifier());
        alert.Send();
      }
      TBWARNING("Hard limit breach in " << m_instrument.GetShortName());
      break;

    default:
      TBERROR("Unknown trading limit state for order " << update);
      break;
    }
  }

  Volume filledVolume;
  bool hasFill = false;
  if (update.HasFilledVolume())
  {
    update.GetFilledVolume(filledVolume);
    double filledVolumeDelta = filledVolume.GetDouble() - m_currentOrderFilledVolume;
    m_totalFilledVolume += filledVolumeDelta;
    m_currentOrderFilledVolume = filledVolume;

    hasFill = (m_tradingInfo.CompareVolumes(filledVolumeDelta, 0) != TB_EQUAL);
  }
  m_order.Merge(update);
  
  Volume activeVolume;
  m_order.GetActiveVolume(activeVolume);
  
  if (GetState() == ManagedOrderState_PendingCreate)
  {
    m_priceInCreateRequest = Price();
    if (activeVolume.GetDouble() > 0)
    {
      SetState(ManagedOrderState_Active);
    }
  }
  else if (GetState() == ManagedOrderState_PendingModify)
  {
    // We have it only when doing synthetic modify
    Boolean deleted;
    if (m_order.HasDeleted() && m_order.GetDeleted(deleted) && deleted)
    {
      SetState(ManagedOrderState_Inactive);
    
      // Since it's now used by the quoter only, let's make quoter decide what to do IF there was a (partial) fill      
      if (!hasFill)
      {
        // Conclude modify if needed
        if (!m_pendingSide.Empty() && !m_pendingVolume.Empty() && !m_pendingPrice.Empty())
        {
          TBDEBUG("Concluding synthetic modify on " << m_pendingSide << " in " << m_instrument.GetShortName());
          Send(m_pendingSide, m_pendingVolume, m_pendingPrice);
        }
        else
        {
          TBSTATUS("Cannot conclude synthetic modify: new order information is already cleared.");
        }
      }
      else
      {
        // Reset pending info and rely on quoter to resend prices since it may re-think quoting
        m_pendingPrice = Price();
        m_pendingVolume = Volume();
        m_pendingSide = Side();
      }

      m_handler.ManagedOrderModule_HandleOrderUpdate(*this, update);
      return;
    }
  }
  
  if (activeVolume.GetDouble() == 0)
  {
    SetState(ManagedOrderState_Inactive);
    m_pendingPrice = Price();
    m_pendingVolume = Volume();
    m_pendingSide = Side();
    m_currentOrderFilledVolume = 0;
  }
  
  if (m_hidden && (m_state == ManagedOrderState_Active) && SharedOrderManager::IsOrderOnMarket(m_order) && !VenueSupportsFAK())
  {
    // Need to pull remains of order if venue doesn't support immediate state
    TBDEBUG("Cancelling remainings of hidden quote since venue doesn't support it.");
    Cancel();
  }

  m_handler.ManagedOrderModule_HandleOrderUpdate(*this, update);
}

void ManagedOrderModule::HandleRequestReply(const Identifier& id, Status status, const String & statusText)
{
  MODULE_DEBUG("ManagedOrderModule::HandleRequestReply()");
  
  if (status == Status::FAIL)
  {
    if (GetState() == ManagedOrderState_PendingCreate)
    {
      TBWARNING("Request " << id << " failed: " << statusText);
      MODULE_STATUS("Order state was: " << GetState());
      SetState(ManagedOrderState_Inactive);
    }
    else
    {
      // this might very well be attempt to modify an already deleted order - do nothing
      TBDEBUG("Request " << id << " failed: " << statusText);
    }
  }
}

bool ManagedOrderModule::CanModifyVolume() const
{
  // When doing two-sided quoting it's unclear how to get an idea when the pending modification is no more pending
  // to be able to correctly tell if there is a potential self-hit situation or not

  return false;

/*  if (m_hidden)
  {
    // Always to delete + insert for hidden orders
    return false;
  }
  return m_ivid.GetVenue().GetTradingCapabilities().CanModify() && m_ivid.GetVenue().GetTradingCapabilities().CanModifyParameter(tbricks::strategy_parameters::ActiveVolume());
*/
}

bool ManagedOrderModule::CanModifyPrice() const
{
  // When doing two-sided quoting it's unclear how to get an idea when the pending modification is no more pending
  // to be able to correctly tell if there is a potential self-hit situation or not
  return false;
/*  if (m_hidden)
  {
    // Always to delete + insert for hidden orders
    return false;
  }
  return m_ivid.GetVenue().GetTradingCapabilities().CanModify() && m_ivid.GetVenue().GetTradingCapabilities().CanModifyParameter(tbricks::strategy_parameters::Price()); 
  */
}

void ManagedOrderModule::SetState(ManagedOrderState newState)
{
  if (newState != m_state)
  {
    MODULE_DEBUG("New state: " << FormatState(newState) << " (old was " << FormatState(m_state) << ")");
    m_state = newState;
    if (newState == ManagedOrderState_Inactive)
    {
      // Clean up order info when becoming inactive
      m_order = Order();
      m_priceInCreateRequest = Price();
    }
  }
}

String ManagedOrderModule::FormatState(ManagedOrderState state)
{
  switch (state)
  {
  case ManagedOrderState_Inactive:      return "inactive";
  case ManagedOrderState_Active:        return "active";
  case ManagedOrderState_PendingCreate: return "pending create";
  case ManagedOrderState_PendingDelete: return "pending delete";
  case ManagedOrderState_PendingModify: return "pending modify";

  default:
    return "UNKNOWN";
  }
}


