// NOTE - code logic inspired by Roger Allen's "Ray Tracing in One Weekend Book Series" 
// https://github.com/RayTracing/raytracing.github.io/blob/release/src/InOneWeekend/material.h
// Original Author from Allen's source is noted as: Peter Shirley <ptrshrl@gmail.com>

#pragma once

#include "../utils/geometry.h"

// “rad‑foam helper” namespace
namespace rf {

using Vec3f = ::radfoam::Vec3f;
using Vec4h = ::radfoam::Vec4h;

// device RNG helper
__device__ inline float randf(uint32_t &state) {
    // same LCG as before, returns [0,1)
    state = 1664525u * state + 1013904223u;
    return float(state & 0x00ffffff) / float(0x01000000);
}

// utilities
__device__ inline Vec3f reflect(const Vec3f& v, const Vec3f& n) {
    return v - 2.f * v.dot(n) * n;
}

__device__ inline bool refract(const Vec3f& v, const Vec3f& n,
                               float ni_over_nt, Vec3f& out) {
    Vec3f uv = v.normalized();
    float dt = uv.dot(n);
    float disc = 1.f - ni_over_nt*ni_over_nt*(1 - dt*dt);
    if (disc > 0.f) {
        out = ni_over_nt*(uv - n*dt) - n*sqrtf(disc);
        return true;
    }
    return false;
}

__device__ inline float schlick(float cosine, float ri) {
    float r0 = (1 - ri) / (1 + ri);
    r0 = r0*r0;
    return r0 + (1 - r0)*powf(1 - cosine, 5.f);
}

__device__ inline Vec3f random_in_unit_sphere(uint32_t& rng_state) {
    Vec3f p;
    do {
        p = 2.f * Vec3f(randf(rng_state), randf(rng_state), randf(rng_state))
            - Vec3f(1,1,1);
       } while (p.squaredNorm() >= 1.f);
    return p;
}

// materials
enum MaterialType : uint8_t { LAMBERT=0, METAL, DIELECTRIC };

struct Material {
    MaterialType type;
    Vec3f        albedo;   // color / tint
    float        fuzz;     // for metal
    float        ref_idx;  // for dielectric
};

struct ScatterRec {
    Vec3f attenuation;
    Vec3f direction;
    bool  did_scatter;
};

__device__ inline ScatterRec scatter(const Material& m,
                                     const Vec3f& in_dir,
                                     const Vec3f& p,
                                     const Vec3f& normal,
                                     uint32_t& rng_state)
{
    ScatterRec rec;
    rec.did_scatter = true;

    if (m.type == LAMBERT) {
        Vec3f tgt = normal + random_in_unit_sphere(rng_state);
        rec.direction   = tgt.normalized();
        rec.attenuation = m.albedo;

    } else if (m.type == METAL) {
        Vec3f refl = reflect(in_dir.normalized(), normal);
        refl += m.fuzz * random_in_unit_sphere(rng_state);
        rec.direction   = refl.normalized();
        rec.attenuation = m.albedo;
        rec.did_scatter = true;
    } else { // DIELECTRIC
        rec.attenuation = Vec3f(1,1,1);
        Vec3f outward_n;
        Vec3f refracted;
        float ni_over_nt, cosine, reflect_prob;

        if (in_dir.dot(normal) > 0.f) {
            outward_n   = -normal;
            ni_over_nt  =  m.ref_idx;
            cosine = m.ref_idx * in_dir.dot(normal) / in_dir.norm();
        } else {
            outward_n   = normal;
            ni_over_nt  = 1.f / m.ref_idx;
            cosine      = -in_dir.dot(normal) / in_dir.norm();
        }

        if (refract(in_dir, outward_n, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, m.ref_idx);
        } else {
            reflect_prob = 1.f;
        }

        if (randf(rng_state) < reflect_prob) {
            rec.direction = reflect(in_dir, normal).normalized();
        } else {
            rec.direction = refracted.normalized();
        }
        rec.did_scatter = true;
    }

    return rec;
}

}