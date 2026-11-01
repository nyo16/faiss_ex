# FaissEx NIF Build
#
# Every setting below is an optional environment variable. This file is the
# single source of their defaults: mix.exs passes nothing, and make inherits
# the caller's environment (elixir_make sets MIX_APP_PATH).
#
#   FAISS_PREFIX    - path to pre-installed FAISS (skips building from source)
#   FAISS_GIT_REPO  - FAISS git repository URL, https:// only (ignored when FAISS_PREFIX is set)
#   FAISS_GIT_REV   - FAISS git revision/tag to build (ignored when FAISS_PREFIX is set)
#   FAISS_GIT_SHA   - commit FAISS_GIT_REV must resolve to; empty skips verification
#   FAISS_OPT_LEVEL - "dd" (runtime SIMD dispatch, default) or "generic"
#   USE_CUDA        - "true" to enable GPU support (CUDA 12+)
#   LIBOMP_PREFIX   - macOS only: path to the libomp keg; auto-detected via
#                     `brew --prefix libomp` when unset
#   MIX_APP_PATH    - set by elixir_make

# The pinned FAISS release. CI derives its cache keys from FAISS_DEFAULT_REV.
FAISS_DEFAULT_REV := v1.15.1
FAISS_DEFAULT_SHA := 75c755c5a1fc854756aaf3bd1131213aeb21d8d1

FAISS_GIT_REPO ?= https://github.com/facebookresearch/faiss.git
FAISS_GIT_REV  ?= $(FAISS_DEFAULT_REV)
# Tags are mutable refs — verify the clone resolves to the expected commit.
# The default SHA only applies to the default rev; a custom FAISS_GIT_REV
# builds unverified unless FAISS_GIT_SHA is provided alongside it.
ifeq ($(FAISS_GIT_REV),$(FAISS_DEFAULT_REV))
FAISS_GIT_SHA ?= $(FAISS_DEFAULT_SHA)
endif
FAISS_GIT_SHA   ?=
USE_CUDA        ?= false
FAISS_OPT_LEVEL ?= dd

# Paths
CACHE_DIR    := $(HOME)/.cache/faiss_ex
FAISS_SRC    := $(CACHE_DIR)/faiss-$(subst /,_,$(FAISS_GIT_REV))
# The source tree is shared per rev; each build configuration gets its own
# build directory, so switching FAISS_OPT_LEVEL or USE_CUDA never reuses a
# stale build.
FAISS_BUILD  := $(FAISS_SRC)/build-$(FAISS_OPT_LEVEL)-$(if $(filter true,$(USE_CUDA)),gpu,cpu)
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

NIF_LDFLAGS := -shared
NIF_LDFLAGS += -L$(LIB_DIR)

# Linux: libfaiss_c finds libfaiss.so only through the RUNPATH CMake gave it
# (the FAISS build directory in ~/.cache), and the default --as-needed drops
# the NIF's own -lfaiss because the NIF calls nothing in it. A release
# copied without that cache then fails to load the NIF. --no-as-needed
# keeps libfaiss.so a direct dependency, loaded from $ORIGIN/lib, so
# libfaiss_c's reference is satisfied by the already-loaded copy.
ifneq ($(UNAME_S),Darwin)
  NIF_LDFLAGS += -Wl,--no-as-needed
endif
NIF_LDFLAGS += -lfaiss_c -lfaiss

# C++ runtime for the transitive libfaiss dependency: LLVM libc++ on macOS,
# GNU libstdc++ elsewhere
ifeq ($(UNAME_S),Darwin)
  NIF_LDFLAGS += -lc++
  NIF_LDFLAGS += -undefined dynamic_lookup
  NIF_LDFLAGS += -Wl,-rpath,@loader_path/lib
else
  NIF_LDFLAGS += -lstdc++
  NIF_LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib'
endif

# FAISS's GPU C API headers include cuda_runtime_api.h / cublas_v2.h, so the
# NIF needs the toolkit's headers (it calls no CUDA function itself; libfaiss
# links the CUDA runtime).
CUDA_HOME ?= /usr/local/cuda
ifeq ($(USE_CUDA),true)
  NIF_CFLAGS += -DFAISS_GPU_ENABLED -I$(CUDA_HOME)/include
endif

# ========== System FAISS vs build from source ==========

ifdef FAISS_PREFIX
  # Use pre-installed FAISS: headers from FAISS_PREFIX/include, libs from FAISS_PREFIX/lib
  NIF_CFLAGS += -I$(FAISS_PREFIX)/include
  SYSTEM_LIB_DIR := $(FAISS_PREFIX)/lib
  BUILD_ID := $(SYSTEM_LIB_DIR)
else
  # Build from source: headers from the cloned source tree
  NIF_CFLAGS += -I$(FAISS_SRC)
  BUILD_ID := $(FAISS_BUILD)

  # FAISS_GIT_REV and FAISS_GIT_REPO reach git and the filesystem, so allow
  # only plain ref/URL characters. strip_chars deletes every character in
  # the word list $(2) from $(1); anything left over is disallowed.
  ALNUM := a b c d e f g h i j k l m n o p q r s t u v w x y z \
           A B C D E F G H I J K L M N O P Q R S T U V W X Y Z \
           0 1 2 3 4 5 6 7 8 9
  strip_chars = $(if $(2),$(call strip_chars,$(subst $(firstword $(2)),,$(1)),$(wordlist 2,$(words $(2)),$(2))),$(1))

  ifneq ($(words $(FAISS_GIT_REV)),1)
    $(error FAISS_GIT_REV must be a single git ref matching ^[A-Za-z0-9._/-]+$$)
  endif
  ifneq ($(call strip_chars,$(FAISS_GIT_REV),$(ALNUM) . _ / -),)
    $(error FAISS_GIT_REV must be a single git ref matching ^[A-Za-z0-9._/-]+$$)
  endif
  ifneq ($(words $(FAISS_GIT_REPO)),1)
    $(error FAISS_GIT_REPO must be a single https:// URL)
  endif
  ifneq ($(patsubst https://%,,$(FAISS_GIT_REPO)),)
    $(error FAISS_GIT_REPO must be an https:// URL)
  endif
  ifneq ($(call strip_chars,$(FAISS_GIT_REPO),$(ALNUM) . _ / - : ~ % + @),)
    $(error FAISS_GIT_REPO contains characters outside [A-Za-z0-9._/:~%+@-])
  endif

  # dd compiles every SIMD variant (AVX2/AVX-512 on x86-64, NEON/SVE on
  # aarch64) into libfaiss and picks one at runtime. FAISS's avx2/avx512/sve
  # levels build separately named libraries (libfaiss_avx2, linked by
  # libfaiss_c_avx2) that this build never installed, so they are rejected
  # rather than silently ignored.
  ifeq ($(filter $(FAISS_OPT_LEVEL),dd generic),)
    $(error FAISS_OPT_LEVEL=$(FAISS_OPT_LEVEL) is not supported: use dd (the default; dispatches AVX2/AVX-512/SVE at runtime) or generic)
  endif

  ifeq ($(FAISS_GIT_SHA),)
    $(warning FAISS_GIT_SHA is empty: FAISS $(FAISS_GIT_REV) builds without commit verification)
  else
    ifneq ($(words $(FAISS_GIT_SHA)),1)
      $(error FAISS_GIT_SHA must be a 40-character hex commit id)
    endif
    ifneq ($(call strip_chars,$(FAISS_GIT_SHA),0 1 2 3 4 5 6 7 8 9 a b c d e f),)
      $(error FAISS_GIT_SHA must be a 40-character hex commit id)
    endif
    ifneq ($(words $(subst 0, 0 ,$(subst 1, 1 ,$(subst 2, 2 ,$(subst 3, 3 ,$(subst 4, 4 ,$(subst 5, 5 ,$(subst 6, 6 ,$(subst 7, 7 ,$(subst 8, 8 ,$(subst 9, 9 ,$(subst a, a ,$(subst b, b ,$(subst c, c ,$(subst d, d ,$(subst e, e ,$(subst f, f ,$(FAISS_GIT_SHA)))))))))))))))))),40)
      $(error FAISS_GIT_SHA must be a 40-character hex commit id)
    endif
  endif
endif

# CMake flags (only used when building from source)
CMAKE_FLAGS := -DFAISS_ENABLE_C_API=ON
CMAKE_FLAGS += -DFAISS_ENABLE_PYTHON=OFF
CMAKE_FLAGS += -DBUILD_TESTING=OFF
CMAKE_FLAGS += -DBUILD_SHARED_LIBS=ON
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE=Release
CMAKE_FLAGS += -DFAISS_OPT_LEVEL=$(FAISS_OPT_LEVEL)

ifeq ($(USE_CUDA),true)
  CMAKE_FLAGS += -DFAISS_ENABLE_GPU=ON
else
  CMAKE_FLAGS += -DFAISS_ENABLE_GPU=OFF
endif

# macOS: AppleClang ships no OpenMP runtime on *either* architecture, so FAISS
# has to be pointed at Homebrew's libomp. Ask brew for the keg rather than
# guessing a prefix — `HOMEBREW_PREFIX ?= ...` silently loses to a stale
# exported value (a common Intel-era dotfile leftover), producing flags that
# point at a prefix containing no libomp. MKL stays off on both arches;
# macOS builds use Accelerate/OpenBLAS.
ifeq ($(UNAME_S),Darwin)
  CMAKE_FLAGS += -DFAISS_ENABLE_MKL=OFF

ifeq ($(origin LIBOMP_PREFIX),undefined)
  LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
endif

# `brew --prefix libomp` prints the keg path even when libomp is not
# installed, so require the library itself to exist.
ifeq ($(wildcard $(LIBOMP_PREFIX)/lib/libomp.dylib),)
  LIBOMP_PREFIX :=
endif

ifeq ($(LIBOMP_PREFIX),)
  $(warning libomp not found: install it with `brew install libomp`, or set LIBOMP_PREFIX. FAISS configure will fail without OpenMP.)
else
  CMAKE_FLAGS += -DOpenMP_ROOT=$(LIBOMP_PREFIX)
  CMAKE_FLAGS += -DCMAKE_CXX_FLAGS="-Xclang -fopenmp -I$(LIBOMP_PREFIX)/include"
  CMAKE_FLAGS += -DCMAKE_EXE_LINKER_FLAGS="-L$(LIBOMP_PREFIX)/lib -lomp"
  CMAKE_FLAGS += -DCMAKE_SHARED_LINKER_FLAGS="-L$(LIBOMP_PREFIX)/lib -lomp"
endif
endif

# OpenMP for the NIF itself (FaissEx.set_omp_threads/1 calls
# omp_set_num_threads). The NIF must link the *same* runtime libfaiss uses:
# two OpenMP runtimes in one process hang or abort (LLVM libomp: "OMP:
# Error #15"), or the setting silently misses FAISS's runtime. On macOS
# that is the LIBOMP_PREFIX libomp FAISS was configured with; on Linux,
# -fopenmp with the default cc (GCC) gives libgomp, which the default g++
# build of FAISS links too. A system FAISS (FAISS_PREFIX) must be shown to
# link exactly that runtime; anything else (Intel libiomp5, MacPorts or
# clang libomp, no OpenMP) — and macOS without libomp — gets
# FAISS_EX_NO_OMP: the NIF links no OpenMP and the setting is ignored.
ifdef FAISS_PREFIX
  ifeq ($(UNAME_S),Darwin)
    FAISS_LINKED_LIBS := $(shell otool -L "$(SYSTEM_LIB_DIR)/libfaiss.dylib" 2>/dev/null)
    NIF_OMP_RUNTIME := $(LIBOMP_PREFIX)/lib/libomp.dylib
  else
    FAISS_LINKED_LIBS := $(shell readelf -d "$(SYSTEM_LIB_DIR)/libfaiss.so" 2>/dev/null)
    NIF_OMP_RUNTIME := [libgomp.so
  endif
  ifeq ($(findstring $(NIF_OMP_RUNTIME),$(FAISS_LINKED_LIBS)),)
    NIF_OMP_DISABLED := true
  endif
endif
ifeq ($(UNAME_S),Darwin)
  ifeq ($(LIBOMP_PREFIX),)
    NIF_OMP_DISABLED := true
  endif
endif

ifeq ($(NIF_OMP_DISABLED),true)
  NIF_CFLAGS += -DFAISS_EX_NO_OMP
else ifeq ($(UNAME_S),Darwin)
  NIF_CFLAGS += -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include
  NIF_LDFLAGS += -L$(LIBOMP_PREFIX)/lib -lomp
else
  NIF_CFLAGS += -fopenmp
  NIF_LDFLAGS += -fopenmp
endif

# Targets
NIF_SO := $(PRIV_DIR)/libfaiss_ex.so

# Build-configuration stamp. The recipe runs on every make but rewrites the
# file only when the configuration changes, so its mtime marks the last
# change. Everything copied or linked into priv/ depends on it: switching to
# an older cached build dir (whose libs have older mtimes than the current
# priv/lib copies) must still re-copy and re-link.
BUILD_CONFIG := $(BUILD_ID) $(NIF_CFLAGS) $(NIF_LDFLAGS)
BUILD_STAMP  := $(PRIV_DIR)/.build_config

.PHONY: all clean verify-faiss FORCE

all: $(NIF_SO)

$(BUILD_STAMP): FORCE
	@mkdir -p "$(PRIV_DIR)"
	@printf '%s\n' '$(subst ','\'',$(BUILD_CONFIG))' > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv -f "$@.tmp" "$@"; fi

FORCE:

ifdef FAISS_PREFIX
# ========== System FAISS path ==========

# Copy system libs to priv/lib/ so the NIF can find them at runtime
$(LIB_DIR)/libfaiss.$(SHARED_EXT): $(BUILD_STAMP)
	@echo "==> Copying system libfaiss from $(SYSTEM_LIB_DIR)..."
	mkdir -p "$(LIB_DIR)"
	cp "$(SYSTEM_LIB_DIR)/libfaiss.$(SHARED_EXT)" "$@"
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libfaiss.$(SHARED_EXT) "$@"
endif

$(LIB_DIR)/libfaiss_c.$(SHARED_EXT): $(LIB_DIR)/libfaiss.$(SHARED_EXT) $(BUILD_STAMP)
	@echo "==> Copying system libfaiss_c from $(SYSTEM_LIB_DIR)..."
	mkdir -p "$(LIB_DIR)"
	cp "$(SYSTEM_LIB_DIR)/libfaiss_c.$(SHARED_EXT)" "$@"
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libfaiss_c.$(SHARED_EXT) "$@"
	install_name_tool -change @rpath/libfaiss.$(SHARED_EXT) @loader_path/libfaiss.$(SHARED_EXT) "$@"
endif

$(NIF_SO): c_src/faiss_ex_nif.c $(LIB_DIR)/libfaiss_c.$(SHARED_EXT) $(LIB_DIR)/libfaiss.$(SHARED_EXT) $(BUILD_STAMP)
	@echo "==> Compiling NIF (using system FAISS at $(FAISS_PREFIX))..."
	mkdir -p "$(PRIV_DIR)"
	$(CC) $(NIF_CFLAGS) $< -o "$@" $(NIF_LDFLAGS)

else
# ========== Build from source path ==========

# Fails when the clone's HEAD commit is not FAISS_GIT_SHA. Run after
# cloning and, through the verify-faiss target, on every make, so a warm
# cache checked out at a different commit fails closed. It checks the
# commit only, not the working tree: local edits to cached sources are not
# detected.
verify_faiss_sha = actual=$$(git -C "$(FAISS_SRC)" rev-parse HEAD) || exit 1; \
  if [ "$$actual" != "$(FAISS_GIT_SHA)" ]; then \
    echo "ERROR: FAISS $(FAISS_GIT_REV) resolved to commit $$actual"; \
    echo "       but FAISS_GIT_SHA expects $(FAISS_GIT_SHA)."; \
    echo "       The upstream tag may have moved — refusing to build."; \
    exit 1; \
  fi

check_cmake_version = v=$$(cmake --version 2>/dev/null | sed -n 's/^cmake version \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p'); \
  set -- $$v; \
  if [ -z "$$1" ] || [ "$$1" -lt 3 ] || { [ "$$1" -eq 3 ] && [ "$$2" -lt 24 ]; }; then \
    echo "ERROR: building FAISS needs CMake 3.24+ (found: $${1:-none}.$${2:-none})."; \
    echo "       Install a newer CMake: pip install cmake, or Kitware's apt repository (https://apt.kitware.com)."; \
    exit 1; \
  fi

check_cuda_version = v=$$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p'); \
  if [ -z "$$v" ] || [ "$$v" -lt 12 ]; then \
    echo "ERROR: USE_CUDA=true needs the CUDA 12+ toolkit with nvcc on PATH (found: $${v:-none})."; \
    echo "       FAISS 1.12.0+ dropped CUDA 11; see the GPU Support section of the README."; \
    exit 1; \
  fi

# Step 1: Clone FAISS and verify the pinned commit
$(FAISS_SRC)/.cloned:
	@echo "==> Cloning FAISS $(FAISS_GIT_REV)..."
	mkdir -p "$(CACHE_DIR)"
	git clone --depth 1 --branch "$(FAISS_GIT_REV)" "$(FAISS_GIT_REPO)" "$(FAISS_SRC)"
ifneq ($(FAISS_GIT_SHA),)
	@( $(verify_faiss_sha) ) || { rm -rf "$(FAISS_SRC)"; exit 1; }
endif
	touch "$@"

verify-faiss: $(FAISS_SRC)/.cloned
ifneq ($(FAISS_GIT_SHA),)
	@$(verify_faiss_sha)
endif

# Step 2: Configure FAISS
$(FAISS_BUILD)/Makefile: $(FAISS_SRC)/.cloned | verify-faiss
	@echo "==> Configuring FAISS ($(notdir $(FAISS_BUILD)))..."
	@$(check_cmake_version)
ifeq ($(USE_CUDA),true)
	@$(check_cuda_version)
endif
	cd "$(FAISS_SRC)" && env -u CFLAGS -u LDFLAGS -u CXXFLAGS cmake -B "$(notdir $(FAISS_BUILD))" $(CMAKE_FLAGS)

# Step 3: Build libfaiss
$(FAISS_BUILD)/faiss/libfaiss.$(SHARED_EXT): $(FAISS_BUILD)/Makefile
	@echo "==> Building libfaiss..."
	env -u CFLAGS -u LDFLAGS -u CXXFLAGS $(MAKE) -C "$(FAISS_BUILD)" -j$(NPROC) faiss

# Step 4: Build libfaiss_c
$(FAISS_BUILD)/c_api/libfaiss_c.$(SHARED_EXT): $(FAISS_BUILD)/faiss/libfaiss.$(SHARED_EXT)
	@echo "==> Building libfaiss_c..."
	env -u CFLAGS -u LDFLAGS -u CXXFLAGS $(MAKE) -C "$(FAISS_BUILD)" -j$(NPROC) faiss_c

# Step 5: Copy libfaiss to priv/lib/
$(LIB_DIR)/libfaiss.$(SHARED_EXT): $(FAISS_BUILD)/faiss/libfaiss.$(SHARED_EXT) $(BUILD_STAMP)
	@echo "==> Installing libfaiss from $(notdir $(FAISS_BUILD))..."
	mkdir -p "$(LIB_DIR)"
	cp "$<" "$@"
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libfaiss.$(SHARED_EXT) "$@"
endif

# Step 6: Copy libfaiss_c to priv/lib/
$(LIB_DIR)/libfaiss_c.$(SHARED_EXT): $(FAISS_BUILD)/c_api/libfaiss_c.$(SHARED_EXT) $(LIB_DIR)/libfaiss.$(SHARED_EXT) $(BUILD_STAMP)
	@echo "==> Installing libfaiss_c from $(notdir $(FAISS_BUILD))..."
	mkdir -p "$(LIB_DIR)"
	cp "$<" "$@"
ifeq ($(UNAME_S),Darwin)
	install_name_tool -id @rpath/libfaiss_c.$(SHARED_EXT) "$@"
	install_name_tool -change @rpath/libfaiss.$(SHARED_EXT) @loader_path/libfaiss.$(SHARED_EXT) "$@"
endif

# Step 7: Compile NIF
$(NIF_SO): c_src/faiss_ex_nif.c $(LIB_DIR)/libfaiss_c.$(SHARED_EXT) $(LIB_DIR)/libfaiss.$(SHARED_EXT) $(BUILD_STAMP) | verify-faiss
	@echo "==> Compiling NIF..."
	mkdir -p "$(PRIV_DIR)"
	$(CC) $(NIF_CFLAGS) $< -o "$@" $(NIF_LDFLAGS)

endif

clean:
	rm -rf "$(PRIV_DIR)/libfaiss_ex.so" "$(BUILD_STAMP)" "$(LIB_DIR)"
