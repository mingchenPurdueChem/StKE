CC=gcc
CFLAGS=-O2 -mavx2 -DDEBUG -Wall -Wunused-variable -Wunused-label -fPIC

ROOTDIR=.
SRCDIR=$(ROOTDIR)/src
BLDDIR=$(ROOTDIR)/build

#JSON_DIR=$(HOME)/local/json-c
INSTALL_DIR=$(HOME)

CFLAGSLIB=-shared

$(BLDDIR):
	@mkdir -p $(BLDDIR)

$(BLDDIR)/ke.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $(SRCDIR)/ke.c -o $@

$(BLDDIR)/ke_io.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $(SRCDIR)/ke_io.c -o $@

$(BLDDIR)/test_io.o: | $(BLDDIR)
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $(SRCDIR)/test_io.c -o $@

$(BLDDIR)/test_io.x: $(BLDDIR)/ke.o $(BLDDIR)/ke_io.o $(BLDDIR)/test_io.o
	$(CC) $(CFLAGS) -o $@ $^ -ljson-c -lm -lcblas -lblas

all: $(BLDDIR)/test_io.x $(BLDDIR)/libke.so $(BLDDIR)/libke.a 

lib: $(BLDDIR)/libke.so $(BLDDIR)/libke.a

$(BLDDIR)/libke.so: $(BLDDIR)/ke.o $(BLDDIR)/ke_io.o
	$(CC) $(CFLAGSLIB) $(CFLAGS) -Wl,-soname,libke.so -o $@ $^ -ljson-c -lm -lcblas -lblas

$(BLDDIR)/libke.a: $(BLDDIR)/ke.o $(BLDDIR)/ke_io.o
	$(AR) rcs $@ $^

clean:
	rm -f $(BLDDIR)/ke.o $(BLDDIR)/test_io.o $(BLDDIR)/ke_io.o
	rm -f $(BLDDIR)/test_io.x

install:
	cp -f $(BLDDIR)/libke.so $(INSTALL_DIR)/lib64
	cp -f $(BLDDIR)/libke.a $(INSTALL_DIR)/lib64
	cp -f $(SRCDIR)/ke.h $(INSTALL_DIR)/include

.PHONY: all
