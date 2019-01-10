CC = g++
DEBUG = -ggdb -D_XOPEN_SOURCE
STD = -std=c++11
ERROR = -Wall -Wextra
THR = -pthread
OPT = -O3
CFLAGS = $(STD) $(THR) $(OPT)
TESTFLAGS = $(ERROR) $(STD) $(THR) $(DEBUG)
LFLAGS = $(ERROR)
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
EXE = dln
TESTEXE = dlntest

all: $(EXE)

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXE)
	
$(TESTEXE): $(OBJECTS)
	$(CC) $(OBJECTS) $(TESTFLAGS) -o $(EXE)
	
%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

run:
	./$(EXE)

clean:
	@rm $(OBJECTS) $(EXE) $(TESTEXE) 2>/dev/null || true
