CC=g++

PROTO= ../../../collectors/proto/
UTILS= ../../../collectors/utils/

INCLUDE=-I/usr/local/boost/ \
		-I/usr/local/include/ \
        -I/usr/local/src/quickfix/include \
		-I $(UTILS) \
		-I $(PROTO)

LIBS= -L/usr/local/src/quickfix/lib/ \
		-L/usr/local/lib \
	  	-L../order_book/ \
      	-lquickfix \
		-lboost_date_time \
		-lboost_program_options \
		-lboost_filesystem \
		-lzmq \
		-lprotobuf \
		-lboost_thread \
		-lprotobuf \
		-lrt

CFLAGS=-Wall -std=c++0x -pthread -O3 
LDFLAGS=-static

PROTOBUF_OBJS = spot_fx_md_1.pb.o \
				capk_globals.pb.o 
				

ifeq ($(dbg),1)
        CFLAGS = -Wall -std=c++0x -pthread -DDEBUG -ggdb -g # -DDEBUG_BOOK
endif

all: protos tick_subscribe

LDFLAGS=-static

tick_subscribe : main.o $(PROTOBUF_OBJS) KTimeUtils.o
	$(CC) $(CFLAGS) main.o $(PROTOBUF_OBJS) KTimeUtils.o $(LIBS) -o $@ 

main.o: tick_subscribe.cpp
	$(CC) -o $@ -c $(INCLUDE) $(CFLAGS) tick_subscribe.cpp 

spot_fx_md_1.pb.o: $(PROTO)/spot_fx_md_1.pb.cc $(PROTO)/spot_fx_md_1.pb.h
	$(CC) -o $@ -c $(INCLUDE) $(CFLAGS) $(PROTO)/spot_fx_md_1.pb.cc 

capk_globals.pb.o: $(PROTO)/capk_globals.pb.cc $(PROTO)/capk_globals.pb.h
	$(CC) -o $@ -c $(INCLUDE) $(CFLAGS) $(PROTO)/capk_globals.pb.cc 

KTimeUtils.o: $(UTILS)/KTimeUtils.cpp $(UTILS)/KTimeUtils.h
	$(CC) -o $@ -c $(INCLUDE) $(CFLAGS) $(UTILS)/KTimeUtils.cpp

protos : 
	cd $(PROTO) && make 

clean :
	rm -f *.o tick_subsribe

