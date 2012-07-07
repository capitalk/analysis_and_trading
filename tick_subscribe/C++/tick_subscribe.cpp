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
    subscriber.connect("tcp://127.0.0.1:9000");
    //subscriber.connect("tcp://127.0.0.1:5271");
    //subscriber.connect("tcp://127.0.0.1:5272");
    //subscriber.connect("tcp://127.0.0.1:5273");
    //const char filter[]  = {10};
    const char* filter = "";
    subscriber.setsockopt(ZMQ_SUBSCRIBE, filter, strlen(filter));
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    capkproto::instrument_bbo bbo;

    while (1) {
        zmq::message_t msg;
        rc = subscriber.recv(&msg);
        assert(rc);
        bbo.ParseFromArray(msg.data(), msg.size());
    //    std::cout << bbo.DebugString() << "\n";
        std::cout << "Symbol   : " << bbo.symbol() << "\n";
        std::cout << "BB MIC   : " << bbo.bb_mic() << "\n";
        std::cout << "BB Price : " << (double)bbo.bb_price() << "\n";
        std::cout << "BB Size  : " << (double)bbo.bb_size() << "\n";

        std::cout << "BA MIC   : " << bbo.ba_mic() << "\n";
        std::cout << "BA Price : " << (double)bbo.ba_price() << "\n";
        std::cout << "BA Size  : " << (double)bbo.ba_size() << "\n";
    }
    google::protobuf::ShutdownProtobufLibrary();

}
