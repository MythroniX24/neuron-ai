# Continuum SLM C++ Inference Engine — Simple Makefile
#
# Build:  make
# Clean:  make clean
# Run:    ./build/continuum build/model.bin --prompt "Hello"

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -march=native -ffast-math -Wall
LDFLAGS  := -lm

# Try to detect and link BLAS for faster matmul
BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null || echo "")
ifneq ($(BLAS_LIBS),)
    CXXFLAGS += -DHAVE_BLAS
    LDFLAGS  += $(BLAS_LIBS)
endif

SRCDIR   := src
INCDIR   := include
BUILDDIR := build

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))
TARGET   := $(BUILDDIR)/continuum

all: $(TARGET)

# Android NDK cross-compile target
android-arm64:
	$(MAKE) CXX=aarch64-linux-android21-clang++ \
		CXXFLAGS="-std=c++17 -O3 -march=armv8-a -ffast-math -Wall" \
		LDFLAGS="-lm -static-libstdc++" \
		TARGET=$(BUILDDIR)/continuum-arm64

# Portable build (no -march=native, runs on any ARM CPU)
portable:
	$(MAKE) CXXFLAGS="-std=c++17 -O2 -ffast-math -Wall" all

$(TARGET): $(OBJECTS) | $(BUILDDIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Built: $(TARGET)"

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean android-arm64
