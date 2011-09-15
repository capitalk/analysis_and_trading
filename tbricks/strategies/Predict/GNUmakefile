#SDK		?= ../../../../sdk
COMMON ?= ../../common
HEADERS += $(COMMON)/*.h
HEADERS += $(SDK)/source/plugin
HEADERS += $(SDK)/source/plugin/shared/*.h
SDK		?= /opt/tbricks/sdk
NAME		?= Eye
//CXXFLAGS	+= -I../Common/$(TBRICKS_SYSTEM)
CXXFLAGS	+= -I../../common
CXXFLAGS	+= -I/opt/boost
CXXFLAGS	+= -I/usr/include/python2.7
CXXFLAGS 	+= -lpython2.7
SOURCES		=$(wildcard ./*.cpp)
//SOURCES		+=$(wildcard ../Common/$(TBRICKS_SYSTEM)/*.cpp)
SOURCES		+=$(wildcard ./Common/*.cpp)
SOURCES		+=$(wildcard .$(SDK)/source/plugin/shared/*.cpp)
CC		= g++

include $(SDK)/makefiles/strategy.make

