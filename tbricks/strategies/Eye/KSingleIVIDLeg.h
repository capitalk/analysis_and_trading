
#ifndef __SINGLE_IVID_LEG__
#define __SINGLE_IVID_LEG__

// TBricks
#include "strategy/API.h"

// Local
#include "SharedOrderManager.h"

// Common
#include "Utils.h"

class KSingleIVIDLeg : protected SharedOrderManager::IManager, 
						protected tbricks::ITimerEventHandler
				//KLeg::IKLegListener
				//BestPriceStream::IHandler
{
	public: 
		class ILegListener 
		{
			public:
				virtual void Leg_HandleTrade(KSingleIVIDLeg& leg, const tbricks::Side& side, const tbricks::Volume& tradeVolume, const tbricks::Price& tradePrice, tbricks::MIC mic) {}; 
				virtual void Leg_HandleRecoveryCompleted(KSingleIVIDLeg& leg);
		};

		KSingleIVIDLeg(const tbricks::Strategy& parentStrategy, 
						KSingleIVIDLeg::ILegListener& listener,
						SharedOrderManager& orderManager,
						const tbricks::InstrumentVenueIdentification& ivid,
						const tbricks::PortfolioIdentifier& portfolio,
						const tbricks::String& name);
		
		virtual ~KSingleIVIDLeg(void);
		// SharedOrderManager::IListener
		virtual void HandleOrderUpdate(const tbricks::Order::Update& update);
		// SharedOrderManager::IManager 
		virtual void HandleRecoveryCompleted();
		virtual void HandleRequestReply(const tbricks::Identifier& requestId, tbricks::Status status, const tbricks::String& statusText);
		virtual bool HandleOrderRecovery(const tbricks::Order::Update& update);

		virtual void HandleTimerEvent(const tbricks::Timer& timer);

		void SendOrder(const tbricks::Volume& volume, const tbricks::Price& price, const tbricks::Side& side);
		void SendOrder(const tbricks::Volume& volume, const tbricks::Price& price, const tbricks::Side& side, const tbricks::Order::Options& options);
		void SetActive(bool active);
		bool IsActive() { return m_isActive; }
		void CancelAll();
		size_t GetNumOrdersOnMarket() const ;
		bool IsOrderOnMarket(const tbricks::Order& o) const ;
		bool IsOrderOnMarket(const tbricks::OrderIdentifier& oid) const ;
		tbricks::MIC GetMIC()  const { return m_mic;}
		tbricks::InstrumentVenueIdentification GetIVID()  const{ return m_ivid;}
		tbricks::InstrumentIdentifier GetInstrumentId()  const{ return m_instrumentId;}
		const tbricks::InstrumentTradingInformation& GetInstrumentTradingInformation() const { return m_ITI; }


	private:
		const tbricks::Strategy& m_strategy;
		SharedOrderManager& m_orderManager;
		KSingleIVIDLeg::ILegListener& m_listener;		
		tbricks::InstrumentVenueIdentification m_ivid;
		tbricks::InstrumentIdentifier m_instrumentId;
		tbricks::Instrument m_instrument;
		tbricks::MIC m_mic;
		tbricks::PortfolioIdentifier m_portfolioId;
		tbricks::String m_name;

		tbricks::InstrumentTradingInformation m_ITI;		
		double m_longVol;
		double m_shortVol;
		bool m_isActive;


		//virtual void HandleBestPrice(const StreamIdentifier& stream, const tbricks::BestPrice& bestPrice);
		//virtual void HandleStreamOpen(const StreamIdentifier& stream);
		//virtual void HandleStreamClose(const StreamIdentifier& stream);
		//virtual void HandleStreamStale(const StreamIdentifier& stream);
		//virtual void HandleSnapshotEnd(const StreamIdentifier& stream);

		OrderIdToOrderInfo m_orders;
		//OrderIdToOrderInfo m_orderUpdates;
		RequestToOrderId m_requestToOrder;
		RequestToOrderId m_deleteRequestToOrder;
		//BestPrice m_bbo;
		//BestPriceStream m_bestPriceStream;		
		void PrintQueueStats(const tbricks::String& string = "");

};

#endif // __SINGLE_IVID_LEG__
