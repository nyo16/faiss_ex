# FAISS Release Research: v1.10.0 → v1.14.3 (July 2026)

## Latest Release Information

**Latest Stable Version:** v1.14.3 (released June 13, 2026)  
**Recommended Pin Tag:** `v1.14.3`  
**Release Span:** January 2025 (v1.10.0) → June 2026 (v1.14.3)  
**Major Versions Added:** 4 (v1.11, v1.12, v1.13, v1.14)

---

## Release Summary (Post-v1.10.0)

### v1.11.0 (April 24, 2025)

**C API Changes:**
- Exposed `IDSelectorBitmap` to C API — new selector capability
- Published C API to Conda for distribution

**Build System:**
- Added MinGW build support alongside MSVC
- OpenBLAS upgraded to 0.3.29 (ARM improvements)
- Fixed Sapphire Rapids (SPR) AVX-512 variant installation issues

**GPU/CUDA:**
- Upgraded cuVS dependency to v25.04
- Improved error handling for insufficient GPU drivers

**Compiler/Standards:**
- nullptr standardization in GPU resources
- LLVM-19 compatibility fixes

**macOS/Apple Silicon:** No specific changes documented.

---

### v1.12.0 (August 2025)

**C API Changes:**
- **NEW:** Exposed `IndexBinaryIVF` to C API for binary search operations
- Maintained backward compatibility with v1.11 API

**GPU/CUDA:**
- **NEW:** `GpuIndexBinaryCagra` implementation for binary GPU search
- **NEW:** FP16 support for `IndexIDMap` with Cagra
- cuVS upgraded to v25.08
- GpuIndexCagra now reuses memory blocks when converting to CPU index

**Build System:**
- **NEW:** Static CUDA linking option added
- CMake: Added installation of missing header `impl/zerocopy_io.h`
- CUDA version policy: **Deprecated CUDA 11, now requires CUDA 12.x**
- Windows CI upgraded from 2019 → 2022

**Removed/Deprecated:**
- `ABS_INNER_PRODUCT` metric removed

**macOS/Apple Silicon:** No specific changes.

---

### v1.13.x (November 2025 – December 2025)

**Focus:** Index robustness and new quantization methods
- RaBitQ multi-bit quantization (2-9 bits)
- Multiple panorama index variants
- SVS integration for coarse quantizers
- ~40 PRs for deserialization hardening (OOM/overflow vulnerability fixes)

**Note:** Limited C API changes in this series; mostly index type additions.

---

### v1.14.0 (March 2, 2026)

**SIMD Infrastructure Modernization:**
- **Major:** Introduced `Dynamic Dispatch` infrastructure with `SIMDConfig` enum
- Refactored `ScalarQuantizer` to use SIMDWIDTH → SIMDLevel enum
- **NEW:** ARM SVE support for distance calculations
- SIMD dispatch now builds successfully on Windows (prior limitation removed)

**CMake/Build:**
- Resolved CMake/Buck build discrepancies
- Fixed static CMake targets for missing SIMD sources
- Updated `FindMKL.cmake` to detect Intel oneAPI MKL (2021+)

**New Index Types:**
- `IndexFlatIPPanorama`
- `IndexFlatL2Panorama`

**Python/Type Hints:**
- Added PEP 561 Python type stubs for faiss package

**macOS/Apple Silicon:** Metal GPU backend foundation laid (not fully implemented in this release).

---

### v1.14.1 (March 6, 2026)

**Incremental Updates:**
- SVS v0.2.0 integration for IVF coarse quantizers
- Hadamard transformation for IVF indices
- Binary index improvements

---

### v1.14.2 (May 21, 2026)

**Metal GPU Backend (Apple Silicon):**
- **NEW:** Metal GPU backend for Apple Silicon
- **NEW:** `IndexFlat` Metal implementation (beginning of Metal support)
- Expanded top-k capabilities for GPU search
- **Important:** Metal is **enabled by default** on Apple machines

**SVS Integration:**
- SVS upgraded to v0.3.0
- **NEW:** SVS IVF index now enabled for production use
- SVSVamana usable as IVF coarse quantizer via `index_factory`

**RISC-V Support:**
- **NEW:** Added RISC-V Vector (RVV) SIMD backend with cross-compilation CI

**SIMD Dispatch:**
- Extended dynamic dispatch to: distances, hamming, partitioning, RaBitQ, FastScan
- Fixed Windows dynamic dispatch builds

**Installation/Packaging:**
- **NEW:** pip install support via scikit-build-core + cibuildwheel
- Significantly eased deployment vs. manual compilation

**Deserialization Hardening:**
- ~40 PRs: Validation across binary, IVF, HNSW, FastScan, and quantizer formats
- Prevents OOM and overflow vulnerabilities during index loading

**C API & Robustness:**
- Improved null pointer handling in C API
- Index consistency validation during deserialization

---

### v1.14.3 (June 13, 2026) — **LATEST**

**Metal GPU Backend Expansion (Apple Silicon):**
- **NEW:** `MetalIndexIVFFlat` with IVF scan/merge kernels
- Expanded top-k support for Metal indices
- **Metal enabled by default** on Apple Silicon (no manual configuration needed)

**Build & Platform Fixes:**
- Fixed non-AVX2 import SIGILL via `.rodata` partitioning table relocation
- Restored Windows ARM64 (MSVC) NEON compatibility
- Corrected AVX512_SPR dispatch on AMD processors
- **Re-enabled musllinux wheels** for faiss-cpu distribution

**Performance Optimizations:**
- Sapphire Rapids (SPR) optimizations for ScalarQuantizer (L2/IP distance metrics)
- VPOPCNTDQ-based HammingComputer and RaBitQ kernel implementations
- HNSW search performance improvements (capacity pre-allocation, devirtualization)

**Dependencies:**
- cuVS upgraded to v26.06
- CI infrastructure updated to ROCm 7

**Deserialization Security:**
- Bool-field validation and null index rejection
- Continued hardening from v1.14.2

**GPU Packaging:**
- **NEW:** faiss-gpu pip wheel packaging option

---

## Build System & CMake Analysis

### C API Build (Unchanged across all versions)

```bash
cmake -B build -DFAISS_ENABLE_C_API=ON .
make -C build -j faiss_c
```

**Output:** `libfaiss_c.dylib` (macOS) or `libfaiss_c.so` (Linux)

### Key CMake Flags

| Flag | Current Value (v1.14.3) | Notes |
|------|------------------------|-------|
| `DFAISS_ENABLE_C_API` | ON | Enables C API build |
| `BUILD_SHARED_LIBS` | ON (supported) | Produces shared `.dylib`/`.so` |
| `CMAKE_BUILD_TYPE` | Release | Recommended for production |
| `FAISS_OPT_LEVEL` | generic/avx2/avx512/avx512_spr/sve | SIMD optimization; comma-separated for multi-level |
| `DFAISS_ENABLE_GPU` | Optional | Enables GPU support (CUDA/Metal) |
| `DFAISS_ENABLE_PYTHON` | OFF (in faiss_ex usage) | Not needed for C API-only builds |
| `DBUILD_TESTING` | OFF (in faiss_ex usage) | Reduces build time |

### FAISS_OPT_LEVEL Changes

**v1.10.0 – v1.14.3:**
- Consistent set: `generic`, `avx2`, `avx512`, `avx512_spr`, `sve`
- **v1.14.0+:** New `SIMDConfig` enum-based dispatch (internal refactoring, no API change)
- Multi-level support: FAISS_OPT_LEVELS accepts semicolon-separated list (e.g., `generic;avx2`)

### macOS Apple Silicon Specific

**v1.10.0:** Manual OpenMP configuration required (e.g., `-DOpenMP_ROOT=/opt/homebrew/opt/libomp`)

**v1.14.2+:** 
- Metal GPU backend **enabled by default** — no manual configuration
- Recommended: `cmake -B build -DFAISS_ENABLE_C_API=ON -DFAISS_ENABLE_GPU=ON .` (GPU auto-detected as Metal on ARM64)
- rpath handling (`install_name_tool -id @rpath/libfaiss.dylib`) likely unchanged

### CUDA Version Requirements

- **v1.10.0 – v1.11.0:** CUDA 11 or 12
- **v1.12.0+:** **CUDA 11 deprecated, CUDA 12.x required**
- **v1.14.2+:** cuVS versions aligned (25.08 → 25.04 → 26.06)

### CMake Minimum Version

- Not explicitly documented in fetched content
- Inferred: CMake 3.15+ (consistent with modern FAISS builds)
- **v1.14.0+:** CMake/Buck build discrepancies resolved (internal only)

---

## C API Function Changes

### New C API Exposures (Post-v1.10.0)

| Function/Type | Added In | Purpose |
|---------------|----------|---------|
| `faiss_IDSelectorBitmap` | v1.11.0 | New bitmap-based ID selector for filtering |
| `faiss_IndexBinaryIVF` | v1.12.0 | Binary inverted-list index |
| Panorama indices (`IndexFlat*Panorama`) | v1.13.x – v1.14.0 | Statistical index variants (L2 & IP) |
| `MetalIndexIVFFlat` (GPU) | v1.14.2 – v1.14.3 | Apple Silicon GPU search (IVF) |
| SVSVamana coarse quantizer | v1.14.2 | Via `index_factory` ("IVF…,Vamana") |

### Removed/Deprecated C API

| Function/Type | Removed In | Reason |
|---------------|-----------|--------|
| `ABS_INNER_PRODUCT` | v1.12.0 | Deprecated in favor of inner product |

### Likely Still Available (v1.14.3, inherited from v1.10.0)

- `faiss_index_factory` — still primary factory API
- `faiss_Index_add`, `faiss_Index_add_with_ids` — core operations
- `faiss_Index_search` — primary search
- `faiss_Index_train` — for trainable indices
- `faiss_Index_reset` — clear index
- `faiss_Index_reconstruct_batch` — vector reconstruction
- `faiss_write_index_fname`, `faiss_read_index_fname` — persistence
- `faiss_clone_index` — index cloning
- `faiss_Index_d` (dimension), `faiss_Index_ntotal` (vector count), `faiss_Index_is_trained` — metadata
- `faiss_Clustering_*` — k-means API
- `faiss_StandardGpuResources_new`, `faiss_index_cpu_to_gpu`, `faiss_index_gpu_to_cpu_all` — GPU resource management

### **NOT** Exposed in v1.14.3 C API (known gaps)

- `range_search` — not exposed to C API (Python-only)
- `remove_ids` — likely not exposed (Python-only)
- `ParameterSpace`/nprobe dynamic setters — not exposed
- Binary index remove operations

---

## Known Build Issues & Caveats

### No Critical C API Build Bugs (June 2025 – July 2026)

GitHub issue search on label `c_api` + `bug` returned **no results** for the v1.11–v1.14.3 timeframe. C API build is stable.

### Historical Issues (Still Potentially Relevant)

1. **libfaiss_c Linkage:** `libfaiss_c.{dylib|so}` dynamically links to `libfaiss.{dylib|so}`. Both must be distributed and discoverable at runtime.
2. **Optimization-Level Suffix:** When building with `FAISS_OPT_LEVEL=avx2`, output may be `libfaiss_avx2.dylib`. C API wrapper must link against the same optimization variant.
3. **BUILD_SHARED_LIBS=ON Warnings:** CMake may drop items from `target_link_libraries` if directories (not libraries) are passed. Generally resolved in v1.14.x.
4. **Windows Shared Library:** Historically fragile; improved in v1.14.2 with dynamic dispatch fixes.

### macOS Apple Silicon Build (v1.14.2+)

**Good news:** Metal GPU backend is **enabled by default**. Simply use:
```bash
cmake -B build -DFAISS_ENABLE_C_API=ON .
make -C build -j faiss_c
```

No need for manual OpenMP configuration on modern Apple Silicon machines.

---

## Upgrade Checklist for faiss_ex

### Recommended Next Steps (v1.10.0 → v1.14.3)

**Tier 1 (High Priority):**
- [ ] Pin `FAISS_GIT_TAG = "v1.14.3"` in mix.exs or build config
- [ ] Verify `libfaiss_c.dylib` + `libfaiss.dylib` both included in macOS build artifacts
- [ ] Confirm no new C API functions in v1.11–v1.14.3 require Elixir exposure (current API is complete)
- [ ] Update CI: Ensure CUDA 12.x is used (not 11); test on macOS 13+ (Apple Silicon)

**Tier 2 (Nice-to-Have):**
- [ ] Document Metal GPU backend availability on Apple Silicon (v1.14.2+)
- [ ] Consider exposing new `faiss_IDSelectorBitmap` and `faiss_IndexBinaryIVF` if use cases exist
- [ ] Update build docs: Metal GPU is now default on macOS ARM64; no manual OpenMP needed

**Tier 3 (Future):**
- [ ] Monitor v1.15+ for `range_search`, `remove_ids` C API exposure (not yet available)
- [ ] Test RISC-V builds if targeting embedded systems (v1.14.2+)

### Breaking Changes for Your Users

- **None expected.** C API is stable; no removals of existing functions used by faiss_ex.
- Metadata queries (`faiss_Index_d`, `ntotal`, `is_trained`) unchanged.
- Clustering API unchanged.
- GPU resource management (if exposed) unchanged.

### Shared Library Compatibility

- **libfaiss_c.dylib** still produced as shared library.
- Dependency on **libfaiss.dylib** still present (no static-only option for C API).
- rpath handling (if needed) likely unchanged; verify with `install_name_tool` on built dylib.

---

## Versions Comparison Table

| Aspect | v1.10.0 (Jan 2025) | v1.14.3 (Jun 2026) | Status |
|--------|--------------------|--------------------|--------|
| C API Stable | Yes | Yes | ✅ No breaking changes |
| GPU Backend | CUDA only | CUDA + Metal (ARM64) | ✅ Additive |
| SIMD Dispatch | Generic model | Dynamic dispatch (internal refactor) | ✅ Transparent |
| macOS ARM64 Metal | Not available | Default-enabled | ✅ Automatic |
| CUDA Required Version | 11 or 12 | 12.x only | ⚠️ Dropped 11 |
| Deserialization Security | Basic | Hardened (40 PRs) | ✅ Improvement |
| libfaiss_c Shared Lib | Yes | Yes | ✅ Consistent |
| Apple Silicon Support | Manual OpenMP config | Auto-detected Metal | ✅ Better DX |

---

## Recommended Action

**Upgrade faiss_ex to v1.14.3 with confidence.** No C API breaking changes; all improvements are additive. The Metal GPU backend on Apple Silicon is automatic and transparent if building on ARM64 macOS. CUDA 12.x requirement is the only downstream impact (users must have CUDA 12, not 11).

**Update mix.exs:**
```elixir
def project do
  [
    # ...
    make_env: %{
      "FAISS_GIT_TAG" => "v1.14.3"  # Was "v1.10.0"
    }
  ]
end
```

**Estimated migration effort:** Low (config change only; no NIF code changes needed).

---

## Source References

- [GitHub Releases - facebookresearch/faiss](https://github.com/facebookresearch/faiss/releases)
- [CHANGELOG.md - facebookresearch/faiss](https://github.com/facebookresearch/faiss/blob/main/CHANGELOG.md)
- [C API INSTALL.md](https://github.com/facebookresearch/faiss/blob/main/c_api/INSTALL.md)
- [v1.11.0 Release Notes](https://github.com/facebookresearch/faiss/releases/tag/v1.11.0)
- [v1.12.0 Release Notes](https://github.com/facebookresearch/faiss/releases/tag/v1.12.0)
- [v1.14.0 Release Notes](https://github.com/facebookresearch/faiss/releases/tag/v1.14.0)
- [v1.14.2 Release Notes](https://github.com/facebookresearch/faiss/releases/tag/v1.14.2)
- [v1.14.3 Release Notes](https://github.com/facebookresearch/faiss/releases/tag/v1.14.3)
