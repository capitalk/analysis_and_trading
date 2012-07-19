import zmq
import proto_objs.spot_fx_md_1_pb2


context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:9000")
#socket.connect("tcp://83.160.107.178:9000")
socket.setsockopt(zmq.SUBSCRIBE, "EUR/USD")
socket.setsockopt(zmq.SUBSCRIBE, "GBP/USD")

while True:
    [topic, contents] = socket.recv_multipart()
    bbo = proto_objs.spot_fx_md_1_pb2.instrument_bbo();
    bbo.ParseFromString(contents);
    print bbo.symbol, bbo.bb_venue_id, bbo.bb_price, bbo.bb_size, "@", bbo.ba_venue_id, bbo.ba_price, bbo.ba_size

