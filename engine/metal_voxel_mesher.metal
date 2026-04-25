#include <metal_stdlib>
using namespace metal;

struct ExpandParams {
    uint sampleSize;
    uint localHeight;
    uint chunkCount;
    uint chunkSize;
    uint worldHeight;
    uint seed;
    uint _pad0;
    uint _pad1;
};

struct MesherParams {
    uint sampleSize;
    uint localHeight;
    uint chunkCount;
    uint chunkSize;
    uint maxVerticesPerChunk;
    float blockScale;
    uint _pad0;
    uint _pad1;
};

struct ChunkCoord {
    int x;
    int y;
    int z;
    int _pad;
};

struct VoxelVertex {
    float4 position;
    float4 normal;
    float4 color;
};

inline uint plane_size(uint sampleSize) {
    return sampleSize * sampleSize;
}

inline uint voxel_index(uint chunkIndex, uint sampleY, uint sampleZ, uint sampleX, uint sampleSize, uint storageHeight) {
    uint plane = plane_size(sampleSize);
    return chunkIndex * (storageHeight * plane) + sampleY * plane + sampleZ * sampleSize + sampleX;
}

inline uint column_linear_index(uint chunkIndex, uint columnX, uint columnZ, uint columnsPerSide) {
    return chunkIndex * (columnsPerSide * columnsPerSide) + columnZ * columnsPerSide + columnX;
}

inline uint mix_u32(uint value) {
    uint x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

inline float hash2(int ix, int iy, uint seed) {
    uint h = uint(ix) * 0x9e3779b9u;
    h = h ^ (uint(iy) * 0x85ebca6bu);
    h = h ^ (seed * 0xc2b2ae35u);
    h = mix_u32(h);
    return float(h & 0x00ffffffu) / 16777215.0f;
}

inline float hash3(int ix, int iy, int iz, uint seed) {
    uint h = uint(ix) * 0x9e3779b9u;
    h = h ^ (uint(iy) * 0x85ebca6bu);
    h = h ^ (uint(iz) * 0xc2b2ae35u);
    h = h ^ (seed * 0x27d4eb2fu);
    h = mix_u32(h);
    return float(h & 0x00ffffffu) / 16777215.0f;
}

inline float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float lerp_f(float a, float b, float t) {
    return a + (b - a) * t;
}

inline float value_noise_2d(float x, float y, uint seed, float frequency) {
    float px = x * frequency;
    float py = y * frequency;
    float x0 = floor(px);
    float y0 = floor(py);
    float xf = px - x0;
    float yf = py - y0;
    int ix0 = int(x0);
    int iy0 = int(y0);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    float v00 = hash2(ix0, iy0, seed);
    float v10 = hash2(ix1, iy0, seed);
    float v01 = hash2(ix0, iy1, seed);
    float v11 = hash2(ix1, iy1, seed);
    float u = fade(xf);
    float v = fade(yf);
    return lerp_f(lerp_f(v00, v10, u), lerp_f(v01, v11, u), v) * 2.0f - 1.0f;
}

inline float value_noise_3d(float x, float y, float z, uint seed, float frequency) {
    float px = x * frequency;
    float py = y * frequency;
    float pz = z * frequency;
    float x0 = floor(px);
    float y0 = floor(py);
    float z0 = floor(pz);
    float xf = px - x0;
    float yf = py - y0;
    float zf = pz - z0;
    int ix0 = int(x0);
    int iy0 = int(y0);
    int iz0 = int(z0);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    int iz1 = iz0 + 1;
    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);
    float c000 = hash3(ix0, iy0, iz0, seed);
    float c100 = hash3(ix1, iy0, iz0, seed);
    float c010 = hash3(ix0, iy1, iz0, seed);
    float c110 = hash3(ix1, iy1, iz0, seed);
    float c001 = hash3(ix0, iy0, iz1, seed);
    float c101 = hash3(ix1, iy0, iz1, seed);
    float c011 = hash3(ix0, iy1, iz1, seed);
    float c111 = hash3(ix1, iy1, iz1, seed);
    float x00 = lerp_f(c000, c100, u);
    float x10 = lerp_f(c010, c110, u);
    float x01 = lerp_f(c001, c101, u);
    float x11 = lerp_f(c011, c111, u);
    return lerp_f(lerp_f(x00, x10, v), lerp_f(x01, x11, v), w) * 2.0f - 1.0f;
}

inline float clamp01(float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline bool should_carve_cave(int worldX, int worldY, int worldZ, int surfaceHeight, uint seed, uint worldHeightLimit) {
    if (worldY <= 3) {
        return false;
    }
    if (worldY >= int(worldHeightLimit) - 2) {
        return false;
    }
    int depthBelowSurface = surfaceHeight - worldY;
    float normalizedY = float(worldY) / float(max(1u, worldHeightLimit - 1u));
    float verticalBand = clamp01(1.0f - abs(normalizedY - 0.45f) * 1.6f);
    if (verticalBand <= 0.0f) {
        return false;
    }
    float xf = float(worldX);
    float yf = float(worldY);
    float zf = float(worldZ);
    constexpr float caveFrequencyScale = 0.5f;
    float cavePrimary = value_noise_3d(xf, yf * 0.85f, zf, seed + 101u, 0.018f * caveFrequencyScale);
    float caveDetail = value_noise_3d(xf, yf * 1.15f, zf, seed + 149u, 0.041666668f * caveFrequencyScale);
    float caveShape = value_noise_3d(xf, yf * 0.35f, zf, seed + 173u, 0.009765625f * caveFrequencyScale);
    float density = cavePrimary * 0.70f + caveDetail * 0.25f - caveShape * 0.10f;
    float depthBonus = min(float(depthBelowSurface) * 0.004f, 0.12f);
    float shallowBonus = depthBelowSurface <= 6 ? (6.0f - float(depthBelowSurface)) * (0.12f / 6.0f) : 0.0f;
    float threshold = 0.62f - verticalBand * 0.08f - depthBonus - shallowBonus;
    if (density > threshold) {
        return true;
    }
    if (depthBelowSurface <= 2) {
        float breachPrimary = value_noise_2d(xf, zf, seed + 211u, 0.020833334f);
        float breachDetail = value_noise_3d(xf, yf, zf, seed + 233u, 0.03125f * caveFrequencyScale);
        float breachDensity = breachPrimary * 0.65f + breachDetail * 0.35f;
        float breachThreshold = 0.78f - verticalBand * 0.06f;
        return breachDensity > breachThreshold;
    }
    return false;
}

inline uint terrain_material_from_surface_profile(int worldX, int worldY, int worldZ, int surfaceHeight, uint surfaceMaterial, uint seed, uint worldHeightLimit) {
    if (worldY < 0 || worldY >= int(worldHeightLimit)) {
        return 0u;
    }
    if (worldY >= surfaceHeight) {
        return 0u;
    }
    if (should_carve_cave(worldX, worldY, worldZ, surfaceHeight, seed, worldHeightLimit)) {
        return 0u;
    }
    if (worldY == 0) {
        return 1u;
    }
    if (worldY < surfaceHeight - 4) {
        return 2u;
    }
    if (worldY < surfaceHeight - 1) {
        return 3u;
    }
    return surfaceMaterial;
}

kernel void expand_surface_to_voxels(
    device const uint       *surfaceHeights    [[buffer(0)]],
    device const uint       *surfaceMaterials  [[buffer(1)]],
    device uint             *blocks            [[buffer(2)]],
    device uint             *voxelMaterials    [[buffer(3)]],
    constant ExpandParams   &params            [[buffer(4)]],
    device const ChunkCoord *chunkCoords       [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint sampleSize = params.sampleSize;
    uint localHeight = params.localHeight;
    uint storageHeight = localHeight + 2u;
    uint chunkCount = params.chunkCount;
    if (gid.x >= sampleSize || gid.y >= sampleSize || gid.z >= chunkCount * storageHeight) {
        return;
    }
    uint chunkIndex = gid.z / storageHeight;
    uint sampleY = gid.z - chunkIndex * storageHeight;
    ChunkCoord coord = chunkCoords[chunkIndex];
    int worldY = coord.y * int(params.chunkSize) + int(sampleY) - 1;
    int worldX = coord.x * int(params.chunkSize) - 1 + int(gid.x);
    int worldZ = coord.z * int(params.chunkSize) - 1 + int(gid.y);
    uint plane = sampleSize * sampleSize;
    uint cellIndex = gid.y * sampleSize + gid.x;
    uint surfaceIndex = chunkIndex * plane + cellIndex;
    uint dstIndex = voxel_index(chunkIndex, sampleY, gid.y, gid.x, sampleSize, storageHeight);
    int surfaceHeight = int(surfaceHeights[surfaceIndex]);
    uint surfaceMaterial = surfaceMaterials[surfaceIndex];
    uint material = terrain_material_from_surface_profile(worldX, worldY, worldZ, surfaceHeight, surfaceMaterial, params.seed, params.worldHeight);
    blocks[dstIndex] = material == 0u ? 0u : 1u;
    voxelMaterials[dstIndex] = material;
}

inline bool solid_at_sample(device const uint *blocks, uint chunkIndex, uint sampleSize, uint storageHeight, int sampleX, int sampleZ, int sampleY) {
    if (sampleX < 0 || sampleZ < 0 || sampleY < 0) {
        return false;
    }
    if (sampleX >= int(sampleSize) || sampleZ >= int(sampleSize) || sampleY >= int(storageHeight)) {
        return false;
    }
    uint idx = voxel_index(chunkIndex, uint(sampleY), uint(sampleZ), uint(sampleX), sampleSize, storageHeight);
    return blocks[idx] != 0u;
}

inline float3 terrain_color(uint height) {
    float altitude = clamp((float(height) - 30.0f) / 56.0f, 0.0f, 1.0f);
    if (height <= 14u) return float3(0.78f, 0.71f, 0.49f);
    if (height >= 90u) return float3(0.95f, 0.97f, 0.98f);
    return mix(float3(0.18f, 0.53f, 0.18f), float3(0.31f, 0.68f, 0.24f), altitude);
}

inline float3 material_color(uint material, uint height) {
    switch (material) {
        case 1u: return float3(0.24f, 0.22f, 0.20f);
        case 2u: return float3(0.42f, 0.40f, 0.38f);
        case 3u: return float3(0.47f, 0.31f, 0.18f);
        case 4u: return mix(float3(0.18f, 0.53f, 0.18f), float3(0.31f, 0.68f, 0.24f), clamp((float(height) - 360.0f) / 1280.0f, 0.0f, 1.0f));
        case 5u: return float3(0.78f, 0.71f, 0.49f);
        case 6u: return float3(0.95f, 0.97f, 0.98f);
        default: return terrain_color(height);
    }
}

inline void emit_vertex(device VoxelVertex *dst, uint slot, float3 p, float3 n, float3 c) {
    dst[slot].position = float4(p, 1.0f);
    dst[slot].normal = float4(n, 0.0f);
    dst[slot].color = float4(c, 1.0f);
}

inline void emit_quad(device VoxelVertex *dst, uint base, float3 p0, float3 p1, float3 p2, float3 p3, float3 normal, float3 color) {
    emit_vertex(dst, base + 0u, p0, normal, color);
    emit_vertex(dst, base + 1u, p1, normal, color);
    emit_vertex(dst, base + 2u, p2, normal, color);
    emit_vertex(dst, base + 3u, p0, normal, color);
    emit_vertex(dst, base + 4u, p2, normal, color);
    emit_vertex(dst, base + 5u, p3, normal, color);
}

inline uint exposed_face_vertex_count(device const uint *blocks, uint chunkIndex, uint sampleSize, uint storageHeight, uint localX, uint localZ, uint localY) {
    uint sampleY = localY + 1u;
    int sx = int(localX);
    int sz = int(localZ);
    int sy = int(sampleY);
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx, sz, sy)) {
        return 0u;
    }
    uint count = 0u;
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx, sz, sy + 1)) count += 6u;
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx, sz, sy - 1)) count += 6u;
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx + 1, sz, sy)) count += 6u;
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx - 1, sz, sy)) count += 6u;
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx, sz + 1, sy)) count += 6u;
    if (!solid_at_sample(blocks, chunkIndex, sampleSize, storageHeight, sx, sz - 1, sy)) count += 6u;
    return count;
}

kernel void count_columns_fixed_slice(
    device const uint     *blocks             [[buffer(0)]],
    device uint           *columnVertexCounts [[buffer(1)]],
    constant MesherParams &params             [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint columnsPerSide = params.sampleSize - 2u;
    if (gid.x >= columnsPerSide || gid.y >= columnsPerSide || gid.z >= params.chunkCount) {
        return;
    }
    uint chunkIndex = gid.z;
    uint localX = gid.x + 1u;
    uint localZ = gid.y + 1u;
    uint storageHeight = params.localHeight + 2u;
    uint columnVertexCount = 0u;
    for (uint y = 0u; y < params.localHeight; ++y) {
        columnVertexCount += exposed_face_vertex_count(blocks, chunkIndex, params.sampleSize, storageHeight, localX, localZ, y);
    }
    columnVertexCounts[column_linear_index(chunkIndex, gid.x, gid.y, columnsPerSide)] = columnVertexCount;
}

kernel void scan_columns_fixed_slice_serial(
    device const uint     *columnVertexCounts  [[buffer(0)]],
    device uint           *columnVertexOffsets [[buffer(1)]],
    device uint           *chunkVertexCounts   [[buffer(2)]],
    device uint           *overflowFlags       [[buffer(3)]],
    constant MesherParams &params              [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.chunkCount) {
        return;
    }
    uint columnsPerSide = params.sampleSize - 2u;
    uint columnsPerChunk = columnsPerSide * columnsPerSide;
    uint base = gid * columnsPerChunk;
    uint running = 0u;
    for (uint i = 0u; i < columnsPerChunk; ++i) {
        uint count = columnVertexCounts[base + i];
        columnVertexOffsets[base + i] = running;
        running += count;
    }
    if (running > params.maxVerticesPerChunk) {
        overflowFlags[gid] = 1u;
        chunkVertexCounts[gid] = 0u;
        return;
    }
    overflowFlags[gid] = 0u;
    chunkVertexCounts[gid] = running;
}

constant uint kScanThreadgroupWidth = 1024u;

kernel void scan_columns_fixed_slice_parallel(
    device const uint     *columnVertexCounts  [[buffer(0)]],
    device uint           *columnVertexOffsets [[buffer(1)]],
    device uint           *chunkVertexCounts   [[buffer(2)]],
    device uint           *overflowFlags       [[buffer(3)]],
    constant MesherParams &params              [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint chunkIndex = tgid.z;
    if (chunkIndex >= params.chunkCount) {
        return;
    }
    uint columnsPerSide = params.sampleSize - 2u;
    uint columnsPerChunk = columnsPerSide * columnsPerSide;
    if (columnsPerChunk > kScanThreadgroupWidth) {
        if (lid.x == 0u) {
            uint base = chunkIndex * columnsPerChunk;
            uint running = 0u;
            for (uint i = 0u; i < columnsPerChunk; ++i) {
                uint count = columnVertexCounts[base + i];
                columnVertexOffsets[base + i] = running;
                running += count;
            }
            if (running > params.maxVerticesPerChunk) {
                overflowFlags[chunkIndex] = 1u;
                chunkVertexCounts[chunkIndex] = 0u;
            } else {
                overflowFlags[chunkIndex] = 0u;
                chunkVertexCounts[chunkIndex] = running;
            }
        }
        return;
    }
    uint tid = lid.x;
    uint base = chunkIndex * columnsPerChunk;
    threadgroup uint scratch[kScanThreadgroupWidth];
    threadgroup uint lastValue;
    uint value = 0u;
    if (tid < columnsPerChunk) {
        value = columnVertexCounts[base + tid];
    }
    scratch[tid] = value;
    if (tid == columnsPerChunk - 1u) {
        lastValue = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1u; offset < kScanThreadgroupWidth; offset <<= 1u) {
        uint index = ((tid + 1u) * offset * 2u) - 1u;
        if (index < kScanThreadgroupWidth) {
            scratch[index] += scratch[index - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        scratch[kScanThreadgroupWidth - 1u] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = kScanThreadgroupWidth >> 1u; offset > 0u; offset >>= 1u) {
        uint index = ((tid + 1u) * offset * 2u) - 1u;
        if (index < kScanThreadgroupWidth) {
            uint temp = scratch[index - offset];
            scratch[index - offset] = scratch[index];
            scratch[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < columnsPerChunk) {
        columnVertexOffsets[base + tid] = scratch[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint running = columnsPerChunk > 0u ? scratch[columnsPerChunk - 1u] + lastValue : 0u;
        if (running > params.maxVerticesPerChunk) {
            overflowFlags[chunkIndex] = 1u;
            chunkVertexCounts[chunkIndex] = 0u;
        } else {
            overflowFlags[chunkIndex] = 0u;
            chunkVertexCounts[chunkIndex] = running;
        }
    }
}

kernel void emit_columns_fixed_slice(
    device const uint       *blocks              [[buffer(0)]],
    device const uint       *materials           [[buffer(1)]],
    device const ChunkCoord *chunkCoords         [[buffer(2)]],
    device const uint       *columnVertexCounts  [[buffer(3)]],
    device const uint       *columnVertexOffsets [[buffer(4)]],
    device const uint       *overflowFlags       [[buffer(5)]],
    device VoxelVertex      *vertexPool          [[buffer(6)]],
    constant MesherParams   &params              [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint columnsPerSide = params.sampleSize - 2u;
    if (gid.x >= columnsPerSide || gid.y >= columnsPerSide || gid.z >= params.chunkCount) {
        return;
    }
    uint chunkIndex = gid.z;
    if (overflowFlags[chunkIndex] != 0u) {
        return;
    }
    uint columnIndex = column_linear_index(chunkIndex, gid.x, gid.y, columnsPerSide);
    uint columnVertexCount = columnVertexCounts[columnIndex];
    if (columnVertexCount == 0u) {
        return;
    }
    uint localX = gid.x + 1u;
    uint localZ = gid.y + 1u;
    uint storageHeight = params.localHeight + 2u;
    uint dstBase = chunkIndex * params.maxVerticesPerChunk + columnVertexOffsets[columnIndex];
    uint written = 0u;
    ChunkCoord coord = chunkCoords[chunkIndex];
    float chunkWorldSize = float(params.chunkSize) * params.blockScale;
    float originX = float(coord.x) * chunkWorldSize;
    float originY = float(coord.y) * chunkWorldSize;
    float originZ = float(coord.z) * chunkWorldSize;
    int sx = int(localX);
    int sz = int(localZ);

    for (uint y = 0u; y < params.localHeight; ++y) {
        uint sampleY = y + 1u;
        int sy = int(sampleY);
        uint idx = voxel_index(chunkIndex, sampleY, localZ, localX, params.sampleSize, storageHeight);
        if (blocks[idx] == 0u) {
            continue;
        }
        uint material = materials[idx];
        float x0 = originX + float(localX - 1u) * params.blockScale;
        float x1 = x0 + params.blockScale;
        float y0 = originY + float(y) * params.blockScale;
        float y1 = y0 + params.blockScale;
        float z0 = originZ + float(localZ - 1u) * params.blockScale;
        float z1 = z0 + params.blockScale;
        uint worldHeightSample = uint(max(0, coord.y * int(params.chunkSize) + int(y)));
        float3 baseColor = material_color(material, worldHeightSample);

        if (!solid_at_sample(blocks, chunkIndex, params.sampleSize, storageHeight, sx, sz, sy + 1)) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y1, z0), float3(x0, y1, z1), float3(x1, y1, z1), float3(x1, y1, z0),
                float3(0.0f, 1.0f, 0.0f), baseColor * 1.00f);
            written += 6u;
        }
        if (!solid_at_sample(blocks, chunkIndex, params.sampleSize, storageHeight, sx, sz, sy - 1)) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0), float3(x1, y0, z0), float3(x1, y0, z1), float3(x0, y0, z1),
                float3(0.0f, -1.0f, 0.0f), baseColor * 0.50f);
            written += 6u;
        }
        if (!solid_at_sample(blocks, chunkIndex, params.sampleSize, storageHeight, sx + 1, sz, sy)) {
            emit_quad(vertexPool, dstBase + written,
                float3(x1, y0, z0), float3(x1, y1, z0), float3(x1, y1, z1), float3(x1, y0, z1),
                float3(1.0f, 0.0f, 0.0f), baseColor * 0.80f);
            written += 6u;
        }
        if (!solid_at_sample(blocks, chunkIndex, params.sampleSize, storageHeight, sx - 1, sz, sy)) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0), float3(x0, y0, z1), float3(x0, y1, z1), float3(x0, y1, z0),
                float3(-1.0f, 0.0f, 0.0f), baseColor * 0.64f);
            written += 6u;
        }
        if (!solid_at_sample(blocks, chunkIndex, params.sampleSize, storageHeight, sx, sz + 1, sy)) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z1), float3(x1, y0, z1), float3(x1, y1, z1), float3(x0, y1, z1),
                float3(0.0f, 0.0f, 1.0f), baseColor * 0.72f);
            written += 6u;
        }
        if (!solid_at_sample(blocks, chunkIndex, params.sampleSize, storageHeight, sx, sz - 1, sy)) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0), float3(x0, y1, z0), float3(x1, y1, z0), float3(x1, y0, z0),
                float3(0.0f, 0.0f, -1.0f), baseColor * 0.60f);
            written += 6u;
        }
    }
}
