import zmq
import proto_objs.interactive_brokers.py


context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5274")
#socket.connect("tcp://83.160.107.178:9000")
# NB - you MUST set a filter for subscribe even if it is just "" empty
socket.setsockopt(zmq.SUBSCRIBE, "")

while True:
    # Use topic/contents for aggregated market receipt - from 9000
    # Use msg only for individual feed subscription from single markets
    #[topic, contents] = socket.recv_multipart()
    msg = socket.recv()
    #print("MSG: [%s]" % (msg))
    bbo = proto_objs.spot_fx_md_1_pb2.interactive_brokers_bbo();
    bbo.ParseFromString(msg);
    print bbo.venue_id, bbo.symbol, bbo.bid_size, bbo.bid_price, "@", bbo.ask_size, bbo.ask_price

