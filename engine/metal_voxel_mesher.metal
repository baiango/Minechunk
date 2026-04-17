#include <metal_stdlib>
using namespace metal;

struct MesherParams {
    uint sampleSize;          // chunkSize + 2
    uint heightLimit;
    uint chunkCount;
    uint chunkSize;           // interior chunk size, e.g. 32
    uint maxVerticesPerChunk; // fixed slice per chunk
    float blockScale;
    uint _pad1;
    uint _pad2;
};

struct ChunkCoord {
    int x;
    int z;
};

struct VoxelVertex {
    float4 position;
    float4 normal;
    float4 color;
};

inline uint voxel_index(
    uint chunkIndex,
    uint y,
    uint localZ,
    uint localX,
    uint sampleSize,
    uint heightLimit
) {
    uint plane = sampleSize * sampleSize;
    return chunkIndex * (heightLimit * plane) + y * plane + localZ * sampleSize + localX;
}

inline float3 terrain_color(uint height) {
    if (height <= 14u) return float3(0.78f, 0.71f, 0.49f);
    if (height >= 90u) return float3(0.95f, 0.97f, 0.98f);
    if (height >= 70u) return float3(0.60f, 0.72f, 0.49f);
    if (height >= 40u) return float3(0.38f, 0.64f, 0.31f);
    return float3(0.28f, 0.54f, 0.22f);
}

inline float3 material_color(uint material, uint height) {
    switch (material) {
        case 1u: return float3(0.24f, 0.22f, 0.20f);
        case 2u: return float3(0.42f, 0.40f, 0.38f);
        case 3u: return float3(0.47f, 0.31f, 0.18f);
        case 4u: return float3(0.31f, 0.68f, 0.24f);
        case 5u: return float3(0.78f, 0.71f, 0.49f);
        case 6u: return float3(0.95f, 0.97f, 0.98f);
        default: return terrain_color(height);
    }
}

inline void emit_vertex(device VoxelVertex *dst, uint slot, float3 p, float3 n, float3 c) {
    dst[slot].position = float4(p, 1.0f);
    dst[slot].normal   = float4(n, 0.0f);
    dst[slot].color    = float4(c, 1.0f);
}

inline void emit_quad(
    device VoxelVertex *dst,
    uint base,
    float3 p0, float3 p1, float3 p2, float3 p3,
    float3 normal,
    float3 c0,
    float3 c1,
    float3 c2,
    float3 c3
) {
    emit_vertex(dst, base + 0u, p0, normal, c0);
    emit_vertex(dst, base + 1u, p1, normal, c1);
    emit_vertex(dst, base + 2u, p2, normal, c2);
    emit_vertex(dst, base + 3u, p0, normal, c0);
    emit_vertex(dst, base + 4u, p2, normal, c2);
    emit_vertex(dst, base + 5u, p3, normal, c3);
}

inline bool solid_at(
    device const uint *blocks,
    uint chunkIndex,
    uint localZ,
    uint localX,
    int sampleY,
    uint sampleSize,
    uint heightLimit
) {
    if (sampleY < 0 || sampleY >= int(heightLimit)) {
        return false;
    }
    return blocks[voxel_index(chunkIndex, uint(sampleY), localZ, localX, sampleSize, heightLimit)] != 0u;
}

inline float ambient_occlusion_factor(bool side1, bool side2, bool corner) {
    uint occlusion = 0u;
    if (side1 && side2) {
        occlusion = 3u;
    } else {
        occlusion = (side1 ? 1u : 0u) + (side2 ? 1u : 0u) + (corner ? 1u : 0u);
    }
    switch (occlusion) {
        case 0u: return 1.0f;
        case 1u: return 0.82f;
        case 2u: return 0.68f;
        default: return 0.54f;
    }
}

inline float ao_y_plane(
    device const uint *blocks,
    uint chunkIndex,
    uint localX,
    uint localZ,
    int sampleY,
    int dx,
    int dz,
    uint sampleSize,
    uint heightLimit
) {
    bool side1 = solid_at(blocks, chunkIndex, localZ, uint(int(localX) + dx), sampleY, sampleSize, heightLimit);
    bool side2 = solid_at(blocks, chunkIndex, uint(int(localZ) + dz), localX, sampleY, sampleSize, heightLimit);
    bool corner = solid_at(blocks, chunkIndex, uint(int(localZ) + dz), uint(int(localX) + dx), sampleY, sampleSize, heightLimit);
    return ambient_occlusion_factor(side1, side2, corner);
}

inline float ao_x_plane(
    device const uint *blocks,
    uint chunkIndex,
    uint sampleX,
    uint localZ,
    int y,
    int dy,
    int dz,
    uint sampleSize,
    uint heightLimit
) {
    bool side1 = solid_at(blocks, chunkIndex, localZ, sampleX, y + dy, sampleSize, heightLimit);
    bool side2 = solid_at(blocks, chunkIndex, uint(int(localZ) + dz), sampleX, y, sampleSize, heightLimit);
    bool corner = solid_at(blocks, chunkIndex, uint(int(localZ) + dz), sampleX, y + dy, sampleSize, heightLimit);
    return ambient_occlusion_factor(side1, side2, corner);
}

inline float ao_z_plane(
    device const uint *blocks,
    uint chunkIndex,
    uint localX,
    uint sampleZ,
    int y,
    int dx,
    int dy,
    uint sampleSize,
    uint heightLimit
) {
    bool side1 = solid_at(blocks, chunkIndex, sampleZ, uint(int(localX) + dx), y, sampleSize, heightLimit);
    bool side2 = solid_at(blocks, chunkIndex, sampleZ, localX, y + dy, sampleSize, heightLimit);
    bool corner = solid_at(blocks, chunkIndex, sampleZ, uint(int(localX) + dx), y + dy, sampleSize, heightLimit);
    return ambient_occlusion_factor(side1, side2, corner);
}

inline uint column_linear_index(uint chunkIndex, uint columnX, uint columnZ, uint columnsPerSide) {
    return chunkIndex * (columnsPerSide * columnsPerSide) + columnZ * columnsPerSide + columnX;
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
    uint plane = params.sampleSize * params.sampleSize;

    uint columnVertexCount = 0u;
    for (uint y = 0u; y < params.heightLimit; ++y) {
        uint idx = voxel_index(chunkIndex, y, localZ, localX, params.sampleSize, params.heightLimit);
        if (blocks[idx] == 0u) continue;

        if (y + 1u >= params.heightLimit || blocks[idx + plane] == 0u) columnVertexCount += 6u;
        if (y == 0u || blocks[idx - plane] == 0u)                     columnVertexCount += 6u;
        if (blocks[idx + 1u] == 0u)                                   columnVertexCount += 6u;
        if (blocks[idx - 1u] == 0u)                                   columnVertexCount += 6u;
        if (blocks[idx + params.sampleSize] == 0u)                    columnVertexCount += 6u;
        if (blocks[idx - params.sampleSize] == 0u)                    columnVertexCount += 6u;
    }

    columnVertexCounts[column_linear_index(chunkIndex, gid.x, gid.y, columnsPerSide)] = columnVertexCount;
}

kernel void scan_columns_fixed_slice_serial(
    device const uint     *columnVertexCounts [[buffer(0)]],
    device uint           *columnVertexOffsets[[buffer(1)]],
    device uint           *chunkVertexCounts  [[buffer(2)]],
    device uint           *overflowFlags      [[buffer(3)]],
    constant MesherParams &params             [[buffer(4)]],
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
    device const uint     *columnVertexCounts [[buffer(0)]],
    device uint           *columnVertexOffsets[[buffer(1)]],
    device uint           *chunkVertexCounts  [[buffer(2)]],
    device uint           *overflowFlags      [[buffer(3)]],
    constant MesherParams &params             [[buffer(4)]],
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
        uint running = 0u;
        if (columnsPerChunk > 0u) {
            running = scratch[columnsPerChunk - 1u] + lastValue;
        }
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
    device const uint       *blocks             [[buffer(0)]],
    device const uint       *materials          [[buffer(1)]],
    device const ChunkCoord *chunkCoords        [[buffer(2)]],
    device const uint       *columnVertexCounts [[buffer(3)]],
    device const uint       *columnVertexOffsets[[buffer(4)]],
    device const uint       *overflowFlags      [[buffer(5)]],
    device VoxelVertex      *vertexPool         [[buffer(6)]],
    constant MesherParams   &params             [[buffer(7)]],
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
    if (columnVertexCount == 0u) return;

    uint localX = gid.x + 1u;
    uint localZ = gid.y + 1u;
    uint plane = params.sampleSize * params.sampleSize;
    uint dstBase = chunkIndex * params.maxVerticesPerChunk + columnVertexOffsets[columnIndex];
    uint written = 0u;

    ChunkCoord cc = chunkCoords[chunkIndex];
    float chunkWorldSize = float(params.chunkSize) * params.blockScale;
    float originX = float(cc.x) * chunkWorldSize;
    float originZ = float(cc.z) * chunkWorldSize;

    for (uint y = 0u; y < params.heightLimit; ++y) {
        uint idx = voxel_index(chunkIndex, y, localZ, localX, params.sampleSize, params.heightLimit);
        if (blocks[idx] == 0u) continue;

        uint material = materials[idx];
        float x0 = originX + float(localX - 1u) * params.blockScale;
        float x1 = x0 + params.blockScale;
        float y0 = float(y) * params.blockScale;
        float y1 = y0 + params.blockScale;
        float z0 = originZ + float(localZ - 1u) * params.blockScale;
        float z1 = z0 + params.blockScale;

        float3 top    = material_color(material, y) * 1.00f;
        float3 bottom = material_color(material, y) * 0.50f;
        float3 east   = material_color(material, y) * 0.80f;
        float3 west   = material_color(material, y) * 0.64f;
        float3 south  = material_color(material, y) * 0.72f;
        float3 north  = material_color(material, y) * 0.60f;
        int yi = int(y);

        if (y + 1u >= params.heightLimit || blocks[idx + plane] == 0u) {
            float ao0 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi + 1, -1, -1, params.sampleSize, params.heightLimit);
            float ao1 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi + 1, 1, -1, params.sampleSize, params.heightLimit);
            float ao2 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi + 1, 1, 1, params.sampleSize, params.heightLimit);
            float ao3 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi + 1, -1, 1, params.sampleSize, params.heightLimit);
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y1, z0),
                float3(x0, y1, z1),
                float3(x1, y1, z1),
                float3(x1, y1, z0),
                float3(0.0f, 1.0f, 0.0f),
                top * ao0,
                top * ao3,
                top * ao2,
                top * ao1);
            written += 6u;
        }

        if (y == 0u || blocks[idx - plane] == 0u) {
            float ao0 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi - 1, -1, -1, params.sampleSize, params.heightLimit);
            float ao1 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi - 1, 1, -1, params.sampleSize, params.heightLimit);
            float ao2 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi - 1, 1, 1, params.sampleSize, params.heightLimit);
            float ao3 = ao_y_plane(blocks, chunkIndex, localX, localZ, yi - 1, -1, 1, params.sampleSize, params.heightLimit);
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0),
                float3(x1, y0, z0),
                float3(x1, y0, z1),
                float3(x0, y0, z1),
                float3(0.0f, -1.0f, 0.0f),
                bottom * ao0,
                bottom * ao3,
                bottom * ao2,
                bottom * ao1);
            written += 6u;
        }

        if (blocks[idx + 1u] == 0u) {
            float ao0 = ao_x_plane(blocks, chunkIndex, localX + 1u, localZ, yi, -1, -1, params.sampleSize, params.heightLimit);
            float ao1 = ao_x_plane(blocks, chunkIndex, localX + 1u, localZ, yi, 1, -1, params.sampleSize, params.heightLimit);
            float ao2 = ao_x_plane(blocks, chunkIndex, localX + 1u, localZ, yi, 1, 1, params.sampleSize, params.heightLimit);
            float ao3 = ao_x_plane(blocks, chunkIndex, localX + 1u, localZ, yi, -1, 1, params.sampleSize, params.heightLimit);
            emit_quad(vertexPool, dstBase + written,
                float3(x1, y0, z0),
                float3(x1, y1, z0),
                float3(x1, y1, z1),
                float3(x1, y0, z1),
                float3(1.0f, 0.0f, 0.0f),
                east * ao0,
                east * ao1,
                east * ao2,
                east * ao3);
            written += 6u;
        }

        if (blocks[idx - 1u] == 0u) {
            float ao0 = ao_x_plane(blocks, chunkIndex, localX - 1u, localZ, yi, -1, -1, params.sampleSize, params.heightLimit);
            float ao1 = ao_x_plane(blocks, chunkIndex, localX - 1u, localZ, yi, -1, 1, params.sampleSize, params.heightLimit);
            float ao2 = ao_x_plane(blocks, chunkIndex, localX - 1u, localZ, yi, 1, 1, params.sampleSize, params.heightLimit);
            float ao3 = ao_x_plane(blocks, chunkIndex, localX - 1u, localZ, yi, 1, -1, params.sampleSize, params.heightLimit);
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0),
                float3(x0, y0, z1),
                float3(x0, y1, z1),
                float3(x0, y1, z0),
                float3(-1.0f, 0.0f, 0.0f),
                west * ao0,
                west * ao1,
                west * ao2,
                west * ao3);
            written += 6u;
        }

        if (blocks[idx + params.sampleSize] == 0u) {
            float ao0 = ao_z_plane(blocks, chunkIndex, localX, localZ + 1u, yi, -1, -1, params.sampleSize, params.heightLimit);
            float ao1 = ao_z_plane(blocks, chunkIndex, localX, localZ + 1u, yi, 1, -1, params.sampleSize, params.heightLimit);
            float ao2 = ao_z_plane(blocks, chunkIndex, localX, localZ + 1u, yi, 1, 1, params.sampleSize, params.heightLimit);
            float ao3 = ao_z_plane(blocks, chunkIndex, localX, localZ + 1u, yi, -1, 1, params.sampleSize, params.heightLimit);
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z1),
                float3(x1, y0, z1),
                float3(x1, y1, z1),
                float3(x0, y1, z1),
                float3(0.0f, 0.0f, 1.0f),
                south * ao0,
                south * ao1,
                south * ao2,
                south * ao3);
            written += 6u;
        }

        if (blocks[idx - params.sampleSize] == 0u) {
            float ao0 = ao_z_plane(blocks, chunkIndex, localX, localZ - 1u, yi, -1, -1, params.sampleSize, params.heightLimit);
            float ao1 = ao_z_plane(blocks, chunkIndex, localX, localZ - 1u, yi, -1, 1, params.sampleSize, params.heightLimit);
            float ao2 = ao_z_plane(blocks, chunkIndex, localX, localZ - 1u, yi, 1, 1, params.sampleSize, params.heightLimit);
            float ao3 = ao_z_plane(blocks, chunkIndex, localX, localZ - 1u, yi, 1, -1, params.sampleSize, params.heightLimit);
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0),
                float3(x0, y1, z0),
                float3(x1, y1, z0),
                float3(x1, y0, z0),
                float3(0.0f, 0.0f, -1.0f),
                north * ao0,
                north * ao1,
                north * ao2,
                north * ao3);
            written += 6u;
        }
    }
}
