CC = g++
STD = -std=c++11
THR = -pthread
OPTDEBUG = -O3
ERROR = -Wall -Wextra
CFLAGS = $(STD) $(THR)
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
EXE = dln

ifdef TEST
OPTDEBUG = -ggdb -D_XOPEN_SOURCE $(ERROR)
endif

ifdef SERVER
CC = g++-8.3.0
endif

all: $(EXE)

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OPTDEBUG) $(OBJECTS) -o $(EXE)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $(OPTDEBUG) $< -o $@

run:
	./$(EXE)

clean:
	@rm $(OBJECTS) $(EXE) 2>/dev/null || true
