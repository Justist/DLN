CC = g++
DEBUG = #-ggdb -D_XOPEN_SOURCE
STD = -std=c++11
ERROR = -Wall -Wextra
THR = -pthread
OPT = -O3
CFLAGS = $(STD) $(THR)
TESTFLAGS = $(ERROR) $(STD) $(THR) $(DEBUG)
LFLAGS = $(ERROR)
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
EXE = dln
TESTEXE = dlntest

all: $(EXE)

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OPT) $(OBJECTS) -o $(EXE)
	
$(TESTEXE): $(OBJECTS)
	$(CC) $(OBJECTS) $(TESTFLAGS) -o $(TESTEXE)
	
%.o: %.cpp
	$(CC) -c $(CFLAGS) $(DEBUG) $< -o $@

run:
	./$(EXE)

clean:
	@rm $(OBJECTS) $(EXE) $(TESTEXE) 2>/dev/null || true
