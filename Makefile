# Makefile for GNU Make

default: run

all: llama.app

run: llama.app
	./llama.app single
	./llama.app half

INCLUDE_COMMON=../../../common
MKL_COPTS = -DMKL_ILP64  -qmkl=sequential
MKL_LIBS  = -lsycl -lOpenCL -lpthread -lm -ldl

app_OPTS = -O2 $(MKL_COPTS) $(MKL_LIBS)

llama.app: run.cpp
	icpx -fsycl -I$(INCLUDE_COMMON) $< -o $@ $(app_OPTS)

clean:
	-rm -f llama.app

.PHONY: clean run all
