#include <zmq.hpp>

#include <iostream>
#include <string>
#include <exception>

#include "spot_fx_md_1.pb.h"


int 
main(int argc, char** argv)
{

    zmq::context_t context(1);
    int rc;
    std::cout << "Connecting to server...\n";
    zmq::socket_t subscriber(context, ZMQ_SUB);
    subscriber.connect("tcp://127.0.0.1:5271");
    subscriber.connect("tcp://127.0.0.1:5272");
    subscriber.connect("tcp://127.0.0.1:5273");
    //const char filter[]  = {10};
    const char* filter = "";
    subscriber.setsockopt(ZMQ_SUBSCRIBE, filter, strlen(filter));
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    capkproto::venue_bbo bbo;

    while (1) {
        zmq::message_t msg;
        rc = subscriber.recv(&msg);
        assert(rc);
        bbo.ParseFromArray(msg.data(), msg.size());
        //std::cout << bbo.DebugString() << "\n";
        std::cout << "Venue : " << bbo.venue() << "\n";
        std::cout << "Name : " << bbo.name() << "\n";
        std::cout << "Bid Size : " << (double)bbo.bid_size() << "\n";
        std::cout << "Bid Price: " << (double)bbo.bid_price() << "\n";
        std::cout << "Ask Size : " << (double)bbo.ask_size() << "\n";
        std::cout << "Ask Price: " << (double)bbo.ask_price() << "\n";
    }
    google::protobuf::ShutdownProtobufLibrary();

}
