PROTOC=protoc
PROTO=../../../collectors/proto/
PROTO_OUT=./

all: protos


protos:
	$(PROTOC) -I$(PROTO) --python_out=$(PROTO_OUT) $(PROTO)/*.proto

clean :
	rm -f *.o collect $(PROTO)/*.cc $(PROTO)/*.h

