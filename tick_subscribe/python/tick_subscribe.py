import zmq
import spot_fx_md_1_pb2


context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:9000")
socket.setsockopt(zmq.SUBSCRIBE, "EUR/USD")
socket.setsockopt(zmq.SUBSCRIBE, "GBP/USD")

while True:
    [topic, contents] = socket.recv_multipart()
    #print("TOPIC: [%s]" % (topic))
    bbo = spot_fx_md_1_pb2.instrument_bbo();
    #strmsg = bbo.SerializeToString()
    bbo.ParseFromString(contents);
    print bbo.symbol, bbo.bb_mic, bbo.bb_price, bbo.bb_size, "@", bbo.ba_mic, bbo.ba_price, bbo.ba_size

