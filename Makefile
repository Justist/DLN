CC = g++
DEBUG = -ggdb -D_XOPEN_SOURCE
STD = -std=c++11
ERROR = -Wall -Wextra
THR = -pthread
OPT = -O3
CFLAGS = $(ERROR) $(STD) $(THR) $(DEBUG) $(OPT)
LFLAGS = $(ERROR)
EXE = simple

all: simple

simple: simpleNetwork.cpp
	g++ $(CFLAGS) -o $(EXE) simpleNetwork.cpp
	
run:
	./$(EXE)

clean:
	@rm $(EXE) 2>/dev/null || true
