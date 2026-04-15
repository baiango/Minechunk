#include <metal_stdlib>
using namespace metal;

struct MesherParams {
    uint sampleSize;          // chunkSize + 2
    uint heightLimit;
    uint chunkCount;
    uint chunkSize;           // interior chunk size, e.g. 32
    uint maxVerticesPerChunk; // fixed slice per chunk
    uint _pad0;
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
    float3 color
) {
    emit_vertex(dst, base + 0u, p0, normal, color);
    emit_vertex(dst, base + 1u, p1, normal, color);
    emit_vertex(dst, base + 2u, p2, normal, color);
    emit_vertex(dst, base + 3u, p0, normal, color);
    emit_vertex(dst, base + 4u, p2, normal, color);
    emit_vertex(dst, base + 5u, p3, normal, color);
}

inline bool reserve_chunk_slice(
    device atomic_uint *chunkVertexCounts,
    device atomic_uint *overflowFlags,
    uint chunkIndex,
    uint need,
    uint maxVerticesPerChunk,
    thread uint &baseOut
) {
    uint expected = atomic_load_explicit(&chunkVertexCounts[chunkIndex], memory_order_relaxed);

    while (true) {
        if (expected > maxVerticesPerChunk || need > (maxVerticesPerChunk - expected)) {
            atomic_store_explicit(&overflowFlags[chunkIndex], 1u, memory_order_relaxed);
            return false;
        }

        uint desired = expected + need;
        if (atomic_compare_exchange_weak_explicit(
                &chunkVertexCounts[chunkIndex],
                &expected,
                desired,
                memory_order_relaxed,
                memory_order_relaxed)) {
            baseOut = expected;
            return true;
        }
        // expected updated on failure
    }
}

kernel void mesh_columns_fixed_slice(
    device const uint       *blocks            [[buffer(0)]],
    device const uint       *materials         [[buffer(1)]],
    device const ChunkCoord *chunkCoords       [[buffer(2)]],
    device atomic_uint      *chunkVertexCounts [[buffer(3)]],
    device atomic_uint      *overflowFlags     [[buffer(4)]],
    device VoxelVertex      *vertexPool        [[buffer(5)]],
    constant MesherParams   &params            [[buffer(6)]],
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

    if (columnVertexCount == 0u) return;

    uint chunkLocalBase = 0u;
    if (!reserve_chunk_slice(
            chunkVertexCounts,
            overflowFlags,
            chunkIndex,
            columnVertexCount,
            params.maxVerticesPerChunk,
            chunkLocalBase)) {
        return;
    }

    uint dstBase = chunkIndex * params.maxVerticesPerChunk + chunkLocalBase;
    uint written = 0u;

    ChunkCoord cc = chunkCoords[chunkIndex];
    float originX = float(cc.x * int(params.chunkSize));
    float originZ = float(cc.z * int(params.chunkSize));

    for (uint y = 0u; y < params.heightLimit; ++y) {
        uint idx = voxel_index(chunkIndex, y, localZ, localX, params.sampleSize, params.heightLimit);
        if (blocks[idx] == 0u) continue;

        uint material = materials[idx];
        float x0 = originX + float(localX - 1u);
        float x1 = x0 + 1.0f;
        float y0 = float(y);
        float y1 = y0 + 1.0f;
        float z0 = originZ + float(localZ - 1u);
        float z1 = z0 + 1.0f;

        float3 top    = material_color(material, y) * 1.00f;
        float3 bottom = material_color(material, y) * 0.50f;
        float3 east   = material_color(material, y) * 0.80f;
        float3 west   = material_color(material, y) * 0.64f;
        float3 south  = material_color(material, y) * 0.72f;
        float3 north  = material_color(material, y) * 0.60f;

        if (y + 1u >= params.heightLimit || blocks[idx + plane] == 0u) {
            // fixed winding for +Y
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y1, z0),
                float3(x0, y1, z1),
                float3(x1, y1, z1),
                float3(x1, y1, z0),
                float3(0.0f, 1.0f, 0.0f),
                top);
            written += 6u;
        }

        if (y == 0u || blocks[idx - plane] == 0u) {
            // fixed winding for -Y
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0),
                float3(x1, y0, z0),
                float3(x1, y0, z1),
                float3(x0, y0, z1),
                float3(0.0f, -1.0f, 0.0f),
                bottom);
            written += 6u;
        }

        if (blocks[idx + 1u] == 0u) {
            emit_quad(vertexPool, dstBase + written,
                float3(x1, y0, z0),
                float3(x1, y1, z0),
                float3(x1, y1, z1),
                float3(x1, y0, z1),
                float3(1.0f, 0.0f, 0.0f),
                east);
            written += 6u;
        }

        if (blocks[idx - 1u] == 0u) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0),
                float3(x0, y0, z1),
                float3(x0, y1, z1),
                float3(x0, y1, z0),
                float3(-1.0f, 0.0f, 0.0f),
                west);
            written += 6u;
        }

        if (blocks[idx + params.sampleSize] == 0u) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z1),
                float3(x1, y0, z1),
                float3(x1, y1, z1),
                float3(x0, y1, z1),
                float3(0.0f, 0.0f, 1.0f),
                south);
            written += 6u;
        }

        if (blocks[idx - params.sampleSize] == 0u) {
            emit_quad(vertexPool, dstBase + written,
                float3(x0, y0, z0),
                float3(x0, y1, z0),
                float3(x1, y1, z0),
                float3(x1, y0, z0),
                float3(0.0f, 0.0f, -1.0f),
                north);
            written += 6u;
        }
    }
}
