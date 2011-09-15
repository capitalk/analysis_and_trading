/*
 *  ManagedOrderModule.h
 *  Strategies
 *
 *  Created by Konstantin Romanov on 7/8/09.
 *  Copyright 2009-2011 Tbricks. All rights reserved.
 *
 */

#ifndef MANAGEDORDERMODULE_H
#define MANAGEDORDERMODULE_H

#include <strategy/API.h>
#include "SharedOrderManager.h"

using namespace tbricks;

class ManagedOrderModule : public SharedOrderManager::IManager
{
public:
  class IModuleHandler
  {
  public:
    virtual void ManagedOrderModule_HandleOrderUpdate(ManagedOrderModule& sender, const Order::Update& update) = 0;
//    virtual void ManagedOrderModule_HandleOrderFill(ManagedOrderModule& sender, const Side& side, const Volume& filledVolume) = 0;
  };
  
  enum ManagedOrderState
  {
    ManagedOrderState_Inactive = 0,
    ManagedOrderState_Active,
    ManagedOrderState_PendingCreate,
    ManagedOrderState_PendingDelete,
    ManagedOrderState_PendingModify
  };
  
  ManagedOrderModule(const Strategy& parentStrategy, 
                     SharedOrderManager& orderManager,
                     const InstrumentVenueIdentification& ivid, 
                     const PortfolioIdentifier& portfolio,
                     ManagedOrderModule::IModuleHandler& handler,
                     const String& name);
  
  virtual ~ManagedOrderModule();
  
  void ModifyPrice(const Price& price);
  void ModifyActiveVolume(const Volume& volume, bool forceUpdateVolume = false);
  void IncreaseActiveVolume(const Volume& volume);
  void DecreaseActiveVolume(const Volume& volume);
  
  void ModifyPriceAndActiveVolume(const Price& price, const Volume& volume, bool forceUpdateVolume = false);
  
  void Send(const Side& side, const Volume& volume, const Price& price);
  void Cancel();
  
  ManagedOrderState GetState() const { return m_state; }
  double GetFilledVolume() const { return m_totalFilledVolume; }
  
  Price GetPrice() const;
  Price GetPendingPrice() const;
  Volume GetActiveVolume() const;

  void SetHidden(bool hidden);
  void SetMinimumVolume(const Volume& volume);
private:
  bool VenueSupportsFAK() const;
  void CreateActivationConditions(const Side& side, const Price& price, Order::ActivationConditions& conditions) const;

  virtual void HandleRecoveryCompleted();
  virtual void HandleOrderUpdate(const Order::Update& update);
  virtual void HandleRequestReply(const Identifier& id, Status status, const String & statusText);
  virtual bool HandleOrderRecovery(const Order::Update& update);
  
  void SetState(ManagedOrderState newState);
  static String FormatState(ManagedOrderState state);

  bool CanModifyVolume() const;
  bool CanModifyPrice() const;

private:
  const Strategy&                     m_strategy;
  SharedOrderManager&                 m_orderManager;
  ManagedOrderModule::IModuleHandler& m_handler;
  String                              m_name;
  InstrumentVenueIdentification       m_ivid;
  PortfolioIdentifier                 m_portfolioId;
  InstrumentTradingInformation        m_tradingInfo;
  Instrument                          m_instrument;
  
  Order                               m_order;
  ManagedOrderState                   m_state;
  double                              m_totalFilledVolume;
  double                              m_currentOrderFilledVolume;
  
  Price                               m_pendingPrice;
  Volume                              m_pendingVolume;
  Side                                m_pendingSide;
  Price                               m_priceInCreateRequest;

  bool                                m_hidden;
  Volume                              m_minimumVolume;
};

#endif // MANAGEDORDERMODULE_H

