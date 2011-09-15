
#ifndef __UTILS_H__
#define __UTILS_H__

#include "strategy/API.h"

/* 
 * Helper classes
 */
// Used to maintain a hash of orderID->OrderInfo when managing individual legs 
// of a trade - @see SingleIVIDLeg and SharedOrderManager
struct OrderInfo
{
	tbricks::Order m_order;
	tbricks::Order::Options m_options;
	tbricks::Price m_basePrice;
	tbricks::Volume m_pendingVolume;
	tbricks::Price m_pendingPrice;
	tbricks::Side m_side;
};

// StreamStatus and StreamState may be helpful when keeping a container of streams
// to determine what state the stream is in. Not fully complete since the 
// class should also contain a set of ClearXXX() methods to unset each of the states 
// if needed
// Also, perhaps a set of IsXXX() function to return flag state
enum StreamState 
{
	STREAM_STATE_NONE = 0x0000,
	STREAM_STATE_OPEN = 0x0001,
	STREAM_STATE_CLOSED = 0x0002,
	STREAM_STATE_SNAPSHOT_DONE = 0x0004,
	STREAM_STATE_STALE = 0x0008
};

class StreamStatus
{
	public:
	StreamStatus():_state(STREAM_STATE_NONE) {};
	unsigned short int Clear() { _state = STREAM_STATE_NONE; return _state;};
	unsigned short int SetOpen() { _state &= STREAM_STATE_OPEN; return _state;};
	unsigned short int SetClosed() { _state &= STREAM_STATE_CLOSED; return _state;};
	unsigned short int SetSnapshotDone() { _state &= STREAM_STATE_SNAPSHOT_DONE; return _state;};
	unsigned short int SetStale() { _state &= STREAM_STATE_STALE; return _state;};

	unsigned short int GetState() { return _state;};

	private:
	unsigned short int _state;


};


/* 
 * Typedefs
 */
typedef tbricks::Hash<tbricks::OrderIdentifier, OrderInfo> OrderIdToOrderInfo;
typedef tbricks::Hash<tbricks::Identifier, tbricks::OrderIdentifier> RequestToOrderId;


/* 
 * Functions
 */
tbricks::Double getValueFactor(const tbricks::InstrumentIdentifier &id);

#endif // __UTILS_H__

