#pragma once

#include <cfloat>
#include <cuda_fp16.h>
#include "../utils/geometry.h"
#include "material.cuh"
#include "pipeline.h"

namespace radfoam {

// --- ANIMATION PATHS ---

// (infinity shape)
__device__ inline Vec3f lemniscate_path(Vec3f center, float time) {
    float a = 0.8f;
    return center + Vec3f{
        a * __sinf(time),
        0.2f * __sinf(time) * __cosf(time),
        1.0f + 0.3f * __cosf(time)
    };
}

__device__ inline Vec3f circular_path(Vec3f center, float time) {
    float r = 0.8f;
    return center + Vec3f{
        r * __cosf(time),
        0.3f,
        r * __sinf(time) + 1.0f
    };
}

__device__ inline Vec3f hop_path(Vec3f center, float time) {
    // square size
    const float r          = 0.6f;
    // how high the hops go
    const float hop_height = 0.3f;
    // total time to go around once
    const float period     = 4.0f;

    float t      = fmodf(time, period);
    float edge_t = t - floorf(t);

    // inverted half‑sine to "jump" up
    float y = -hop_height * __sinf(3.14159265f * edge_t);

    float x, z;
    if (t < 1.0f) {
        x = -r + 2*r*edge_t;  z = -r;
    }
    else if (t < 2.0f) {
        x =  r;  z = -r + 2*r*edge_t;
    }
    else if (t < 3.0f) {
        x =  r - 2*r*edge_t;  z =  r;
    }
    else {
        x = -r;  z =  r - 2*r*edge_t;
    }

    return center + Vec3f{ x, y, z };
}

__device__ inline Vec3f animated_sphere_position(float time, int path_type, Vec3f center) {
    if (path_type == 0) return center;
    if (path_type == 1) return lemniscate_path(center, time);
    if (path_type == 2) return circular_path(center, time);
    if (path_type == 3) return hop_path(center, time);
    // else return default as static (just the center)
    return center;
}


template <int block_size, int chunk_size, typename CellFunctor>
__forceinline__ __device__ uint32_t
trace(const Ray &ray,
      const Vec3f *__restrict__ points,
      const uint32_t *__restrict__ point_adjacency,
      const uint32_t *__restrict__ point_adjacency_offsets,
      const Vec4h *__restrict__ adjacent_points,
      uint32_t start_point,
      uint32_t max_steps,
      CellFunctor cell_functor) {
    float t_0 = 0.0f;
    uint32_t n = 0;

    uint32_t current_point_idx = start_point;
    Vec3f primal_point = points[current_point_idx];

    for (;;) {
        n++;
        if (n > max_steps) {
            break;
        }

        // Outer loop iterates through Voronoi cells
        uint32_t point_adjacency_begin =
            point_adjacency_offsets[current_point_idx];
        uint32_t point_adjacency_end =
            point_adjacency_offsets[current_point_idx + 1];

        uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
        float t_1 = std::numeric_limits<float>::infinity();

        uint32_t next_face = UINT32_MAX;
        Vec3f next_point = Vec3f::Zero();

        half2 chunk[chunk_size * 2];
        for (uint32_t i = 0; i < num_faces; i += chunk_size) {
#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j) {
                chunk[2 * j] = reinterpret_cast<const half2 *>(
                    adjacent_points + point_adjacency_begin + i + j)[0];
                chunk[2 * j + 1] = reinterpret_cast<const half2 *>(
                    adjacent_points + point_adjacency_begin + i + j)[1];
            }

#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j) {
                Vec3f offset(__half2float(chunk[2 * j].x),
                             __half2float(chunk[2 * j].y),
                             __half2float(chunk[2 * j + 1].x));
                Vec3f face_origin = primal_point + offset / 2.0f;
                Vec3f face_normal = offset;
                float dp = face_normal.dot(ray.direction);
                float t = (face_origin - ray.origin).dot(face_normal) / dp;

                if (dp > 0.0f && t < t_1 && (i + j) < num_faces) {
                    t_1 = t;
                    next_face = i + j;
                }
            }
        }

        if (next_face == UINT32_MAX) {
            break;
        }

        uint32_t next_point_idx =
            point_adjacency[point_adjacency_begin + next_face];
        next_point = points[next_point_idx];

        if (t_1 > t_0) {
            if (!cell_functor(
                    current_point_idx, t_0, t_1, primal_point, next_point)) {
                break;
            }
        }
        t_0 = fmaxf(t_0, t_1);
        current_point_idx = next_point_idx;
        primal_point = next_point;
    }

    return n;
}

__forceinline__ __device__ Vec3f cell_intersection_grad(
    const Vec3f &primal_point, const Vec3f &opposite_point, const Ray &ray) {
    Vec3f face_origin = (primal_point + opposite_point) / 2.0f;
    Vec3f face_normal = (opposite_point - primal_point);

    float num = (face_origin - ray.origin).dot(face_normal);
    float dp = face_normal.dot(ray.direction);

    Vec3f grad = num * ray.direction + dp * (ray.origin - primal_point);
    grad /= dp * dp;

    return grad;
}

inline RADFOAM_HD uint32_t make_rgba8(float r, float g, float b, float a) {
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));
    a = std::max(0.0f, std::min(1.0f, a));
    int ri = static_cast<int>(r * 255.0f);
    int gi = static_cast<int>(g * 255.0f);
    int bi = static_cast<int>(b * 255.0f);
    int ai = static_cast<int>(a * 255.0f);
    return (ai << 24) | (bi << 16) | (gi << 8) | ri;
}

inline __device__ Vec3f colormap(float v,
                                 ColorMap map,
                                 const CMapTable &cmap_table) {
    int map_len = cmap_table.sizes[map];
    const Vec3f *map_vals =
        reinterpret_cast<const Vec3f *>(cmap_table.data[map]);

    int i0 = static_cast<int>(v * (map_len - 1));
    int i1 = i0 + 1;
    float t = v * (map_len - 1) - i0;
    i0 = max(0, min(i0, map_len - 1));
    i1 = max(0, min(i1, map_len - 1));
    return map_vals[i0] * (1.0f - t) + map_vals[i1] * t;
}

// function for determining when a ray hits the sphere
__device__ bool intersect_sphere(
    const Ray& ray,
    const Vec3f& center,
    float radius,
    float& t_hit,
    Vec3f& normal_out)
{
    Vec3f oc = ray.origin - center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) return false;

    float sqrt_d = sqrtf(discriminant);
    float t0 = (-b - sqrt_d) / (2.0f * a);
    float t1 = (-b + sqrt_d) / (2.0f * a);

    t_hit = (t0 > 0.0f) ? t0 : t1;
    if (t_hit <= 0.0f) return false;

    Vec3f hit_point = ray.origin + t_hit * ray.direction;
    normal_out = (hit_point - center).normalized();
    return true;
}

// Simple 32‑bit integer hash for seeding
// https://stackoverflow.com/questions/17035441/looking-for-decent-quality-prng-with-only-32-bits-of-state
__device__ inline uint32_t wangHash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

// **Unified** ray tracer!
template <typename attr_scalar, int sh_deg>
__device__ Vec3f trace_ray_scene(
    Ray                     ray,
    const SceneSphere&      sphere,
    const Vec3f*            points,
    const attr_scalar*      attrs,
    const uint32_t*         adj,
    const uint32_t*         adj_off,
    const Vec4h*            adj_diff,
    uint32_t                start_point,
    const TraceSettings&    settings,
    const Eigen::VectorXf&  sh_coeffs)
{
    // Ambient background color
    const Vec3f ambient_color(0.01f, 0.01f, 0.02f);

    Vec3f radiance   = Vec3f::Zero();
    Vec3f throughput = Vec3f(1.0f, 1.0f, 1.0f);

    // Seed
    uint32_t seed      = __float_as_uint(ray.origin.x()*12.9898f + ray.origin.y()*78.233f);
    uint32_t rng_state = wangHash(seed);

    for (int bounce = 0; bounce < settings.max_bounces; ++bounce) {
        // Animate sphere center
        Vec3f animated_center = animated_sphere_position(settings.time, sphere.path_type, sphere.center);

        // Try sphere intersection
        float  t_hit;
        Vec3f  n;
        if (intersect_sphere(ray, animated_center, sphere.radius, t_hit, n)) {
            Vec3f hit_point = ray.origin + t_hit * ray.direction;

            if (sphere.material.type == rf::DIELECTRIC) {
                // For the sake of this, treat glass as a perfect mirror without recursion
                Vec3f reflect_dir = rf::reflect(ray.direction.normalized(), n).normalized();
                ray.origin    = hit_point + n * 1e-4f;
                ray.direction = reflect_dir;
                continue;
            }

            // Non‑dielectric fallback
            rf::ScatterRec rec = rf::scatter(
                sphere.material,
                ray.direction,
                hit_point,
                n,
                rng_state
            );
            if (!rec.did_scatter) {
                // black for now, change later?
                break;
            }
            throughput   = throughput.cwiseProduct(rec.attenuation);
            ray.origin   = hit_point + n * 1e-4f;
            ray.direction = rec.direction;
            continue;
        }

        // No sphere hit! Just volume march
        float  trans   = 1.0f;
        Vec3f vol_rgb  = Vec3f::Zero();

        auto cell_functor = [&](uint32_t idx, float t0, float t1,
                                const Vec3f &p0, const Vec3f &p1) {
            constexpr int sh_dim = 3 * (1 + sh_deg)*(1 + sh_deg);
            constexpr int A      = 1 + sh_dim;
            const attr_scalar* ap = attrs + idx * A;

            float s   = float(ap[A-1]);
            Vec3f rgb = (s > 1e-6f)
                      ? load_sh_as_rgb<attr_scalar,sh_deg>(sh_coeffs, ap)
                      : Vec3f::Zero();

            float dt    = fmaxf(t1 - t0, 0.0f);
            float alpha = 1.0f - expf(-s * dt);

            vol_rgb += trans * alpha * rgb;
            trans    = trans * (1.0f - alpha);
            return (trans > settings.weight_threshold);
        };

        trace<128,4>(ray,
                     points,
                     adj,
                     adj_off,
                     adj_diff,
                     start_point,
                     settings.max_intersections,
                     cell_functor);

        if (vol_rgb.norm() > 0.f) {
            radiance += throughput.cwiseProduct(vol_rgb);
        } else {
            radiance += throughput.cwiseProduct(ambient_color);
        }
        break;
    }

    return radiance;
}

} // namespace radfoam