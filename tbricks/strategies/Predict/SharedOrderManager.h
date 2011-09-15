#ifndef SharedOrderManager_H
#define SharedOrderManager_H

#include "strategy/API.h"


class SharedOrderManager : protected tbricks::OrderManager, protected tbricks::IRequestReplyHandler, protected tbricks::ITimerEventHandler
{
public:
  class IOrderListener
  {
  public:
    virtual void HandleOrderUpdate(const tbricks::Order::Update& update) = 0;
  };

  class IManager : public IOrderListener
  {
  public:
    virtual void HandleRecoveryCompleted() = 0;
    virtual void HandleRequestReply(const tbricks::Identifier& requestId, tbricks::Status status, const tbricks::String & statusText) = 0;
    virtual bool HandleOrderRecovery(const tbricks::Order::Update& update) = 0;
  };
  
  SharedOrderManager(void);
  virtual ~SharedOrderManager(void);

  bool StartRecovery(const tbricks::InitializationReason & reason);
  bool InRecovery() const { return OrderManager::InRecovery(); }

  void AddManager(IManager& manager);
  void RemoveManager(IManager& manager);

  void AddListener(IOrderListener& listener);
  void RemoveListener(IOrderListener& listener);

  OrderCreateRequestResult SendCreateRequest(IManager& manager, 
                                             const tbricks::Order::Options& options);
  //OrderCreateRequestResult SendCreateRequest(OrderHandler& handler, const tbricks::Order::Options& options);


  tbricks::Identifier SendModifyRequest(IManager& manager, 
                               const tbricks::OrderIdentifier& orderId,
                               const tbricks::Order::Modifier & modifier);
  tbricks::Identifier SendModifyRequest(const tbricks::OrderIdentifier& orderId,
                               const tbricks::Order::Modifier & modifier);

  tbricks::Identifier SendDeleteRequest(IManager& manager, 
                               const tbricks::OrderIdentifier & orderId);
  tbricks::Identifier SendDeleteRequest(const tbricks::OrderIdentifier& orderId);

  static bool IsOrderOnMarket(const tbricks::Order& order);
private:
  virtual void HandleRecoveryCompleted();
  virtual void HandleOrderUpdate(const tbricks::Order::Update& update);
  virtual void HandleRequestReply(const tbricks::Identifier& id, tbricks::Status status, const tbricks::String & statusText);

  virtual void HandleTimerEvent(const tbricks::Timer& timer);

  enum RequestType
  {
    RequestType_Create,
    RequestType_Modify,
    RequestType_Delete
  };
// KTK - should combine the RequestToManager and RequestToType 
  typedef tbricks::Hash<tbricks::Identifier, IManager*> RequestToManager;
  typedef tbricks::Hash<tbricks::Identifier, RequestType> RequestToType;
  typedef tbricks::Hash<tbricks::OrderIdentifier, IManager*> OrderToManager;
  typedef std::set<IManager*> Managers;
  typedef std::set<IOrderListener*> Listeners;
  
  void MapRequest(const tbricks::Identifier& requestId, IManager& manager, RequestType requestType);
  void MapOrder(const tbricks::OrderIdentifier& orderId, IManager& manager);
  
private:
  RequestToManager  m_requestToManager;
  RequestToType     m_requestToType;
  OrderToManager    m_orderToManager;
  Managers          m_managers;
  Listeners         m_listeners;

  tbricks::Timer             m_internalStatusTimer;

  tbricks::Hash<tbricks::Identifier, tbricks::Order>  m_deleteOldChildrenRequests;
  std::vector<tbricks::Order>  m_recoveryQueue;
  std::vector<tbricks::Order>  m_ordersToDeleteOnRecoveryComplete;

  tbricks::Identifier        m_tracingId;

  friend class Trace;
};

#endif // SharedOrderManager_H


