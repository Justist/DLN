CC = g++
DEBUG = -ggdb
STD = -std=c++11
ERROR = -Wall -Wextra
CFLAGS = $(ERROR) $(STD) $(DEBUG)
LFLAGS = $(ERROR)
EXE = dln
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

$(EXE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(EXE)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@
	
run:
	./$(EXE)

clean:
	@rm $(OBJECTS) $(EXE) 2>/dev/null || true
