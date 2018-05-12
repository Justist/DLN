CC = g++
DEBUG = -ggdb -D_XOPEN_SOURCE
STD = -std=c++11
ERROR = -Wall -Wextra
OPT = -O3
CFLAGS = $(ERROR) $(STD) $(DEBUG) $(OPT)
LFLAGS = $(ERROR)
EXE = dln
SEXE = simple
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXE)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@
	
simple: simpleNetwork.cpp
	g++ $(CFLAGS) -pthread -o $(SEXE) simpleNetwork.cpp
	
run:
	./$(EXE)

clean:
	@rm $(OBJECTS) $(EXE) $(SEXE) 2>/dev/null || true
