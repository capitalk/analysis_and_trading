#include "SharedOrderManager.h"

SharedOrderManager::SharedOrderManager(void):
m_internalStatusTimer(*this)
{
  TBDEBUG("SharedOrderManager::SharedOrderManager()");

  m_internalStatusTimer.Start(tbricks::TimeInterval::Minutes(1));
}

SharedOrderManager::~SharedOrderManager(void)
{
  TBDEBUG("SharedOrderManager::~SharedOrderManager()");
}

bool SharedOrderManager::StartRecovery(const tbricks::InitializationReason & reason)
{
  TBSTATUS("SharedOrderManager::StartRecovery(" << reason << ")");

  bool shouldExpectOrders = tbricks::OrderManager::StartRecovery(reason);
  if (shouldExpectOrders)
  {
    TBSTATUS("Strategy is expected to have orders to recover.");
  }
  else
  {
    TBSTATUS("Strategy does not have orders to recover.");
  }
  return shouldExpectOrders;
}

void SharedOrderManager::AddManager(IManager& manager)
{
  TBDEBUG("SharedOrderManager::AddManager()");
  
  m_managers.insert(&manager);
}

/* 
 * Remove a manager, all related requests, and all releated orders
 */
void SharedOrderManager::RemoveManager(IManager& manager)
{
  TBDEBUG("SharedOrderManager::RemoveManager()");

  Managers::iterator i = m_managers.find(&manager);
  if (i != m_managers.end())
  {
    m_managers.erase(i);
  }
  else
  {
    TBERROR("SharedOrderManager: attempting to remove non-existing manager.");
  }

  std::vector<tbricks::Identifier> requestsToUnmap;
  for (RequestToManager::const_iterator r = m_requestToManager.begin(); r != m_requestToManager.end(); ++r)
  {
    if (r->second == &manager)
    {
      requestsToUnmap.push_back(r->first);
    }
  }
  for (size_t n = 0; n < requestsToUnmap.size(); n++)
  {
    m_requestToType.erase(requestsToUnmap[n]);
    m_requestToManager.erase(requestsToUnmap[n]);
  }

  std::vector<tbricks::OrderIdentifier> ordersToUnmap;
  for (OrderToManager::const_iterator o = m_orderToManager.begin(); o != m_orderToManager.end(); ++o)
  {
    if (o->second == &manager)
    {
      ordersToUnmap.push_back(o->first);
    }
  }
  for (size_t n = 0; n < ordersToUnmap.size(); n++)
  {
    m_orderToManager.erase(ordersToUnmap[n]);
  }
}

void SharedOrderManager::AddListener(IOrderListener& listener)
{
  TBDEBUG("SharedOrderManager::AddListener()");

  m_listeners.insert(&listener);
}

void SharedOrderManager::RemoveListener(IOrderListener& listener)
{
  TBDEBUG("SharedOrderManager::RemoveListener()");

  Listeners::iterator i = m_listeners.find(&listener);
  if (i != m_listeners.end())
  {
    m_listeners.erase(i);
  }
  else
  {
    TBERROR("SharedOrderManager: attempting to remove non-existing listener.");
  }
}

tbricks::OrderManager::OrderCreateRequestResult SharedOrderManager::SendCreateRequest(IManager& manager, const tbricks::Order::Options& options)
{
  TBDEBUG("SharedOrderManager::SendCreateRequest()");
  
  tbricks::Order::Options newOptions(options);
  if (!m_tracingId.Empty())
  {
    newOptions.SetTracingIdentifier(m_tracingId);
  }

  OrderCreateRequestResult result = tbricks::OrderManager::SendCreateRequest(newOptions, *this);
  MapRequest(result.GetRequestIdentifier(), manager, RequestType_Create);
  MapOrder(result.GetOrderIdentifier(), manager);
  
  return result;
}

tbricks::Identifier SharedOrderManager::SendModifyRequest(IManager& manager, const tbricks::OrderIdentifier& orderId, const tbricks::Order::Modifier & modifier)
{
  TBDEBUG("SharedOrderManager::SendModifyRequest()");

  tbricks::Order::Modifier newModifier(modifier);
  if (!m_tracingId.Empty())
  {
    newModifier.SetTracingIdentifier(m_tracingId);
  }
  tbricks::Identifier requestId = tbricks::OrderManager::SendModifyRequest(orderId, newModifier, *this);
  MapRequest(requestId, manager, RequestType_Modify);
  
  return requestId;
}

tbricks::Identifier SharedOrderManager::SendDeleteRequest(IManager& manager, const tbricks::OrderIdentifier & order_id)
{
  TBDEBUG("SharedOrderManager::SendDeleteRequest()");

  tbricks::Order::DeleteOptions options;
  if (!m_tracingId.Empty())
  {
    options.SetTracingIdentifier(m_tracingId);
  }
  tbricks::Identifier requestId = tbricks::OrderManager::SendDeleteRequest(order_id, options, *this);
  MapRequest(requestId, manager, RequestType_Delete);
  
  return requestId;
}

void SharedOrderManager::HandleRecoveryCompleted()
{
  TBDEBUG("SharedOrderManager::HandleRecoveryCompleted()");

  TBSTATUS("OrderManager recovery completed. Notifying individual managers.");
  
  
  for (Managers::iterator i = m_managers.begin(); i != m_managers.end(); i++)
  {
    (*i)->HandleRecoveryCompleted();
  }

  for (size_t n = 0; n < m_ordersToDeleteOnRecoveryComplete.size(); n++)
  {
    tbricks::Identifier requestId = tbricks::OrderManager::SendDeleteRequest(m_ordersToDeleteOnRecoveryComplete[n].GetIdentifier(), *this);
    m_deleteOldChildrenRequests[requestId] = m_ordersToDeleteOnRecoveryComplete[n];

    TBSTATUS("Deleting leftover active order " << m_ordersToDeleteOnRecoveryComplete[n]);
  }
}

void SharedOrderManager::HandleOrderUpdate(const tbricks::Order::Update& update)
{
  TBDEBUG("SharedOrderManager::HandleOrderUpdate()");

	TBDEBUG("COUNT OF MANAGERS: " << m_managers.size());
  bool deleted = false;
  if (update.HasDeleted())
  {
    tbricks::Boolean deletedFlag;
    update.GetDeleted(deletedFlag);
    deleted = deletedFlag;
  }

  if (!InRecovery() || !deleted)
  {
    for (Listeners::iterator l = m_listeners.begin(); l != m_listeners.end(); l++)
    {
      (*l)->HandleOrderUpdate(update);
    }
  }

  if (InRecovery())
  {
    // Kill everything alive on recovery
    if (!deleted)
    {
      TBSTATUS("Scheduling to delete leftover active order " << update);

      m_ordersToDeleteOnRecoveryComplete.push_back(tbricks::Order(update));
    }

    tbricks::Volume filledVolume;
      // Skip all orders which are non filled, but completely deleted
    if (!deleted || (update.GetFilledVolume(filledVolume) && (filledVolume > 0)))
    {

		// KTK - queue the orders for later recovery
		// NOT DONE YET - just make sure that we pause before upload 
		m_recoveryQueue.push_back(tbricks::Order(update));
	
		TBDEBUG("====> Looking for managers for recovered order - there are: " << m_managers.size());
      for (Managers::iterator i = m_managers.begin(); i != m_managers.end(); i++)
      {
		// KTK - handle orders that have a filled volume but may be deleted
        if ((*i)->HandleOrderRecovery(update))
        {
			// KTK - handle orders that have filled volume and are not deleted (partialed orders still on market)
			// and re-assign them to the proper listener/manager
          if (!deleted)
          {
            MapOrder(update.GetIdentifier(), *(*i));
          }
          return;
        }
      }    
    
      TBERROR("SharedOrderManager: order " << update << " is not recovered by any manager.");
    }
  }
  else
  {
    OrderToManager::iterator managerOfOrder = m_orderToManager.find(update.GetIdentifier());

    if (managerOfOrder != m_orderToManager.end())
    {
      managerOfOrder->second->HandleOrderUpdate(update);
    }
    else
    {
      tbricks::Boolean isDeleted;
      if (update.HasDeleted() && update.GetDeleted(isDeleted) && isDeleted.GetBool())
      {
        TBDEBUG("Received update from already deleted order. Skipping.");
      }
      else
      {
        TBERROR("OrderManager: Can't find manager for order update " << update);
      }
    }
  }

  if (deleted)
  {
    TBDEBUG("Order " << update.GetIdentifier() << " is deleted, unmapping.");
    m_orderToManager.erase(update.GetIdentifier());
  }
}

void SharedOrderManager::HandleRequestReply(const tbricks::Identifier& requestId, tbricks::Status status, const tbricks::String & statusText)
{
  TBDEBUG("SharedOrderManager::HandleRequestReply()");
  
  RequestToManager::iterator i = m_requestToManager.find(requestId);
  if (i != m_requestToManager.end())
  {
    RequestToType::iterator j = m_requestToType.find(requestId);
    if (status == tbricks::Status::FAIL)
    {
      tbricks::String requestType;
      switch (j->second)
      {
      case RequestType_Create:
        requestType = "Create";
        break;

      case RequestType_Modify:
        requestType = "Modify";
        break;

      case RequestType_Delete:
        requestType = "Delete";
      }
      TBWARNING(requestType << " request " << requestId << " failed: " << statusText);
    }
    i->second->HandleRequestReply(requestId, status, statusText);
    m_requestToManager.erase(i);
    m_requestToType.erase(j);
  }
  else
  {
    tbricks::Hash<tbricks::Identifier, tbricks::Order>::iterator j = m_deleteOldChildrenRequests.find(requestId);
    if (j != m_deleteOldChildrenRequests.end())
    {
      if (status == tbricks::Status::FAIL)
      {
        TBWARNING("Cannot cancel leftover order on recovery: " << statusText << " : " << j->second);
        tbricks::Alert alert("Cannot cancel leftover order on recovery.", tbricks::Alert::SeverityCritical);
        alert.Send();
      }
      m_deleteOldChildrenRequests.erase(j);
    }
    else
    {
      TBERROR("OrderManager: Can't find manager for request " << requestId << " status: " << status << " text: " << statusText);
    }
  }
}

void SharedOrderManager::HandleTimerEvent(const tbricks::Timer& timer)
{
  TBDUMP("SharedOrderManager::HandleTimerEvent()");

  TBDUMP("SharedOrderManager: m_orderToManager: " << m_orderToManager.size() << "  m_requestToManager: " << m_requestToManager.size() << 
    " m_requestToType: " << m_requestToType.size() << " m_listeners: " << m_listeners.size() << " m_managers: " << m_managers.size() <<
    " m_deleteOldChildrenRequests: " << m_deleteOldChildrenRequests.size() << " m_ordersToDeleteOnRecoveryComplete: " << m_ordersToDeleteOnRecoveryComplete.size());
}

void SharedOrderManager::MapRequest(const tbricks::Identifier& requestId, IManager& manager, RequestType requestType)
{
  m_requestToManager[requestId] = &manager;
  m_requestToType[requestId] = requestType;

}

void SharedOrderManager::MapOrder(const tbricks::OrderIdentifier& orderId, IManager& manager)
{
  m_orderToManager[orderId] = &manager;
}

bool SharedOrderManager::IsOrderOnMarket(const tbricks::Order& order)
{
  if (!order.HasExchangeOrderIdentifier())
  {
    return false;
  }
  if (order.HasDeleted())
  {
    tbricks::Boolean deleted;
    order.GetDeleted(deleted);
    if (deleted)
    {
      return false;
    }
  }
  return true;
}


