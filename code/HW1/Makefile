SRCS = $(wildcard *.c)

PROGS = $(patsubst %.c,%,$(SRCS))

CC = gcc

CFLAGS = -lm -O3

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f $(PROGS)
