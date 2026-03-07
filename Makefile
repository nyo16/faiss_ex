# FaissEx NIF Build
#
# Environment variables (set via mix.exs make_env):
#   FAISS_GIT_REPO - FAISS git repository URL
#   FAISS_GIT_REV  - FAISS git revision/tag to build
#   USE_CUDA       - "true" to enable GPU support
#   MIX_APP_PATH   - set by elixir_make

FAISS_GIT_REPO ?= https://github.com/facebookresearch/faiss.git
FAISS_GIT_REV  ?= v1.10.0
USE_CUDA       ?= false

# Paths
CACHE_DIR    := $(HOME)/.cache/faiss_ex
FAISS_SRC    := $(CACHE_DIR)/faiss-$(subst /,_,$(FAISS_GIT_REV))
FAISS_BUILD  := $(FAISS_SRC)/build
PRIV_DIR     := $(MIX_APP_PATH)/priv
LIB_DIR      := $(PRIV_DIR)/lib

# Erlang
ERTS_INCLUDE_DIR ?= $(shell erl -noshell -eval 'io:format("~ts/erts-~ts/include", [code:root_dir(), erlang:system_info(version)])' -s init stop)

# Platform
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  NPROC := $(shell sysctl -n hw.ncpu)
  SHARED_EXT := dylib
else
  NPROC := $(shell nproc)
  SHARED_EXT := so
endif

# NIF Compiler flags (prefixed to avoid leaking into cmake)
CC ?= cc
NIF_CFLAGS := -std=c11 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
NIF_CFLAGS += -I$(ERTS_INCLUDE_DIR)
NIF_CFLAGS += -I$(FAISS_SRC)

NIF_LDFLAGS := -shared
NIF_LDFLAGS += -L$(LIB_DIR)
NIF_LDFLAGS += -lfaiss_c -lfaiss
NIF_LDFLAGS += -lstdc++

ifeq ($(UNAME_S),Darwin)
  NIF_LDFLAGS += -undefined dynamic_lookup
  NIF_LDFLAGS += -Wl,-rpath,@loader_path/lib
else
  NIF_LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib'
endif

ifeq ($(USE_CUDA),true)
  NIF_CFLAGS += -DFAISS_GPU_ENABLED
endif

# CMake flags
CMAKE_FLAGS := -DFAISS_ENABLE_C_API=ON
CMAKE_FLAGS += -DFAISS_ENABLE_PYTHON=OFF
CMAKE_FLAGS += -DBUILD_TESTING=OFF
CMAKE_FLAGS += -DBUILD_SHARED_LIBS=ON
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Release

ifeq ($(USE_CUDA),true)
  CMAKE_FLAGS += -DFAISS_ENABLE_GPU=ON
else
  CMAKE_FLAGS += -DFAISS_ENABLE_GPU=OFF
endif

# macOS Apple Silicon: no MKL, need OpenMP hints
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
  CMAKE_FLAGS += -DFAISS_ENABLE_MKL=OFF
  HOMEBREW_PREFIX ?= /opt/homebrew
  CMAKE_FLAGS += -DOpenMP_ROOT=$(HOMEBREW_PREFIX)/opt/libomp
  CMAKE_FLAGS += -DCMAKE_CXX_FLAGS="-Xclang -fopenmp -I$(HOMEBREW_PREFIX)/opt/libomp/include"
  CMAKE_FLAGS += -DCMAKE_EXE_LINKER_FLAGS="-L$(HOMEBREW_PREFIX)/opt/libomp/lib -lomp"
  CMAKE_FLAGS += -DCMAKE_SHARED_LINKER_FLAGS="-L$(HOMEBREW_PREFIX)/opt/libomp/lib -lomp"
endif
endif

# Targets
NIF_SO := $(PRIV_DIR)/libfaiss_ex.so

.PHONY: all clean

all: $(NIF_SO)

# Step 1: Clone FAISS
$(FAISS_SRC)/.cloned:
	@echo "==> Cloning FAISS $(FAISS_GIT_REV)..."
	mkdir -p $(CACHE_DIR)
	git clone --depth 1 --branch $(FAISS_GIT_REV) $(FAISS_GIT_REPO) $(FAISS_SRC)
	touch $@

# Step 2: Configure FAISS
$(FAISS_BUILD)/Makefile: $(FAISS_SRC)/.cloned
	@echo "==> Configuring FAISS..."
	cd $(FAISS_SRC) && env -u CFLAGS -u LDFLAGS -u CXXFLAGS cmake -B build $(CMAKE_FLAGS)

# Step 3: Build libfaiss
$(FAISS_BUILD)/faiss/libfaiss.$(SHARED_EXT): $(FAISS_BUILD)/Makefile
	@echo "==> Building libfaiss..."
	env -u CFLAGS -u LDFLAGS -u CXXFLAGS $(MAKE) -C $(FAISS_BUILD) -j$(NPROC) faiss

# Step 4: Build libfaiss_c
$(FAISS_BUILD)/c_api/libfaiss_c.$(SHARED_EXT): $(FAISS_BUILD)/faiss/libfaiss.$(SHARED_EXT)
	@echo "==> Building libfaiss_c..."
	env -u CFLAGS -u LDFLAGS -u CXXFLAGS $(MAKE) -C $(FAISS_BUILD) -j$(NPROC) faiss_c

# Step 5: Copy libfaiss to priv/lib/
$(LIB_DIR)/libfaiss.$(SHARED_EXT): $(FAISS_BUILD)/faiss/libfaiss.$(SHARED_EXT)
	@echo "==> Installing libfaiss..."
	mkdir -p $(LIB_DIR)
	cp $< $@
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libfaiss.$(SHARED_EXT) $@
endif

# Step 6: Copy libfaiss_c to priv/lib/
$(LIB_DIR)/libfaiss_c.$(SHARED_EXT): $(FAISS_BUILD)/c_api/libfaiss_c.$(SHARED_EXT) $(LIB_DIR)/libfaiss.$(SHARED_EXT)
	@echo "==> Installing libfaiss_c..."
	mkdir -p $(LIB_DIR)
	cp $< $@
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libfaiss_c.$(SHARED_EXT) $@
	install_name_tool -change @rpath/libfaiss.$(SHARED_EXT) @loader_path/libfaiss.$(SHARED_EXT) $@
endif

# Step 7: Compile NIF
$(NIF_SO): c_src/faiss_ex_nif.c $(LIB_DIR)/libfaiss_c.$(SHARED_EXT) $(LIB_DIR)/libfaiss.$(SHARED_EXT)
	@echo "==> Compiling NIF..."
	mkdir -p $(PRIV_DIR)
	$(CC) $(NIF_CFLAGS) $< -o $@ $(NIF_LDFLAGS)

clean:
	rm -rf $(PRIV_DIR)/libfaiss_ex.so $(LIB_DIR)
