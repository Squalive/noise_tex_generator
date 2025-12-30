#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, Mul};
use noise::NoiseFn;

#[derive(Debug, Clone)]
pub struct NoiseTextureDescriptor<C: Channels> {
    pub size: [u32; 3],
    channel_configs: [ChannelConfig; 4],
    channel_swizzles: ChannelSwizzles,
    channels: C,
}

impl Default for NoiseTextureDescriptor<()> {
    fn default() -> Self {
        Self {
            size: [1, 1, 1],
            channel_configs: Default::default(),
            channel_swizzles: ChannelSwizzles::default(),
            channels: (),
        }
    }
}

impl<C: Channels> NoiseTextureDescriptor<C> {
    pub fn with_size(mut self, size: [u32; 3]) -> Self {
        self.size = size;
        self
    }

    pub fn with_channel_swizzles(mut self, channel_swizzles: ChannelSwizzles) -> Self {
        self.channel_swizzles = channel_swizzles;
        self
    }

    pub fn with_r<R, RC>(self, r: RC) -> NoiseTextureDescriptor<ChannelR<R>>
    where
        R: NoiseFn<f64, 3>,
        RC: NoiseChannel<R>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            channel_configs: [
                ChannelConfig::new(r.bounds(), r.seamless()),
                Default::default(),
                Default::default(),
                Default::default(),
            ],
            channel_swizzles: self.channel_swizzles,
            channels: ChannelR(r.into_noise()),
        }
    }

    pub fn with_rg<R, G, RC, GC>(self, r: RC, g: GC) -> NoiseTextureDescriptor<ChannelRg<R, G>>
    where
        R: NoiseFn<f64, 3>,
        RC: NoiseChannel<R>,
        G: NoiseFn<f64, 3>,
        GC: NoiseChannel<G>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            channel_configs: [
                ChannelConfig::new(r.bounds(), r.seamless()),
                ChannelConfig::new(g.bounds(), g.seamless()),
                Default::default(),
                Default::default(),
            ],
            channel_swizzles: self.channel_swizzles,
            channels: ChannelRg(r.into_noise(), g.into_noise()),
        }
    }

    pub fn with_rgb<R, G, B, RC, GC, BC>(
        self,
        r: RC,
        g: GC,
        b: BC,
    ) -> NoiseTextureDescriptor<ChannelRgb<R, G, B>>
    where
        R: NoiseFn<f64, 3>,
        RC: NoiseChannel<R>,
        G: NoiseFn<f64, 3>,
        GC: NoiseChannel<G>,
        B: NoiseFn<f64, 3>,
        BC: NoiseChannel<B>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            channel_configs: [
                ChannelConfig::new(r.bounds(), r.seamless()),
                ChannelConfig::new(g.bounds(), g.seamless()),
                ChannelConfig::new(b.bounds(), b.seamless()),
                Default::default(),
            ],
            channel_swizzles: self.channel_swizzles,
            channels: ChannelRgb(r.into_noise(), g.into_noise(), b.into_noise()),
        }
    }

    pub fn with_rgba<R, G, B, A, RC, GC, BC, AC>(
        self,
        r: RC,
        g: GC,
        b: BC,
        a: AC,
    ) -> NoiseTextureDescriptor<ChannelRgba<R, G, B, A>>
    where
        R: NoiseFn<f64, 3>,
        RC: NoiseChannel<R>,
        G: NoiseFn<f64, 3>,
        GC: NoiseChannel<G>,
        B: NoiseFn<f64, 3>,
        BC: NoiseChannel<B>,
        A: NoiseFn<f64, 3>,
        AC: NoiseChannel<A>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            channel_configs: [
                ChannelConfig::new(r.bounds(), r.seamless()),
                ChannelConfig::new(g.bounds(), g.seamless()),
                ChannelConfig::new(b.bounds(), b.seamless()),
                ChannelConfig::new(a.bounds(), a.seamless()),
            ],
            channel_swizzles: self.channel_swizzles,
            channels: ChannelRgba(
                r.into_noise(),
                g.into_noise(),
                b.into_noise(),
                a.into_noise(),
            ),
        }
    }

    pub fn to_texture(&self) -> Vec<u8> {
        #[derive(Clone, Copy)]
        struct PerChannelValues {
            bounds: ChannelBounds,
            x_extent: f64,
            y_extent: f64,
            z_extent: f64,
            x_step: f64,
            y_step: f64,
            z_step: f64,
        }

        struct SeamlessValues {
            sample_coords: [[f64; 3]; 8],
            x_blend: f64,
            y_blend: f64,
            z_blend: f64,
        }

        impl PerChannelValues {
            fn new(width: u32, height: u32, depth: u32, bounds: ChannelBounds) -> Self {
                let x_extent = bounds.x.1 - bounds.x.0;
                let y_extent = bounds.y.1 - bounds.y.0;
                let z_extent = bounds.z.1 - bounds.z.0;

                let x_step = x_extent / width as f64;
                let y_step = y_extent / height as f64;
                let z_step = z_extent / depth as f64;

                Self {
                    bounds,
                    x_extent,
                    y_extent,
                    z_extent,
                    x_step,
                    y_step,
                    z_step,
                }
            }

            fn get_xyz(&self, x: u32, y: u32, z: u32) -> [f64; 3] {
                [
                    self.bounds.x.0 + self.x_step * x as f64,
                    self.bounds.y.0 + self.y_step * y as f64,
                    self.bounds.z.0 + self.z_step * z as f64,
                ]
            }

            fn get_seamless_values(&self, x: f64, y: f64, z: f64) -> SeamlessValues {
                SeamlessValues {
                    sample_coords: [
                        [x, y, z],
                        [x + self.x_extent, y, z],
                        [x, y + self.y_extent, z],
                        [x + self.x_extent, y + self.y_extent, z],
                        [x, y, z + self.z_extent],
                        [x + self.x_extent, y, z + self.z_extent],
                        [x, y + self.y_extent, z + self.z_extent],
                        [x + self.x_extent, y + self.y_extent, z + self.z_extent],
                    ],
                    x_blend: 1.0 - (x - self.bounds.x.0) / self.x_extent,
                    y_blend: 1.0 - (y - self.bounds.y.0) / self.y_extent,
                    z_blend: 1.0 - (z - self.bounds.z.0) / self.z_extent,
                }
            }
        }

        /// Performs linear interpolation between two values.
        #[inline(always)]
        fn lerp<T>(a: T, b: T, alpha: f64) -> T
        where
            T: Mul<f64, Output = T> + Add<Output = T>,
        {
            b * alpha + a * (1.0 - alpha)
        }

        #[inline]
        fn sample_seamless<T: Channels>(
            channels: &T,
            channel: PerChannelValues,
            xyz: [f64; 3],
            sample_fn: impl Fn(&T, [f64; 3]) -> f64,
        ) -> f64 {
            let value = channel.get_seamless_values(xyz[0], xyz[1], xyz[2]);

            // Batch sample all 8 points using array
            let samples: [f64; 8] = [
                sample_fn(channels, value.sample_coords[0]),
                sample_fn(channels, value.sample_coords[1]),
                sample_fn(channels, value.sample_coords[2]),
                sample_fn(channels, value.sample_coords[3]),
                sample_fn(channels, value.sample_coords[4]),
                sample_fn(channels, value.sample_coords[5]),
                sample_fn(channels, value.sample_coords[6]),
                sample_fn(channels, value.sample_coords[7]),
            ];

            // Extract channel values - all stack allocated
            let c000 = samples[0];
            let c100 = samples[1];
            let c010 = samples[2];
            let c110 = samples[3];
            let c001 = samples[4];
            let c101 = samples[5];
            let c011 = samples[6];
            let c111 = samples[7];

            // X interpolation
            let c00 = lerp(c000, c100, value.x_blend);
            let c01 = lerp(c010, c110, value.x_blend);
            let c10 = lerp(c001, c101, value.x_blend);
            let c11 = lerp(c011, c111, value.x_blend);

            // Y interpolation
            let c0 = lerp(c00, c01, value.y_blend);
            let c1 = lerp(c10, c11, value.y_blend);

            // Z interpolation
            lerp(c0, c1, value.z_blend)
        }

        let [width, height, depth] = self.size;

        let mut data = vec![0u8; (width * height * depth) as usize * C::channel_count()];

        if let Some(len) = data.len().checked_div(C::channel_count()) {
            let r_channel =
                PerChannelValues::new(width, height, depth, self.channel_configs[0].bounds);
            let g_channel =
                PerChannelValues::new(width, height, depth, self.channel_configs[1].bounds);
            let b_channel =
                PerChannelValues::new(width, height, depth, self.channel_configs[2].bounds);
            let a_channel =
                PerChannelValues::new(width, height, depth, self.channel_configs[3].bounds);

            let wh = width * height;
            for index in 0..len {
                let z = index as u32 / wh;
                let slice_index = index as u32 % wh;
                let y = slice_index / width;
                let x = slice_index % width;

                let r_xyz = r_channel.get_xyz(x, y, z);
                let g_xyz = g_channel.get_xyz(x, y, z);
                let b_xyz = b_channel.get_xyz(x, y, z);
                let a_xyz = a_channel.get_xyz(x, y, z);

                let r = if self.channel_configs[0].seamless {
                    sample_seamless(&self.channels, r_channel, r_xyz, Channels::get_r)
                } else {
                    self.channels.get_r(r_xyz)
                };

                let g = if self.channel_configs[1].seamless {
                    sample_seamless(&self.channels, g_channel, g_xyz, Channels::get_g)
                } else {
                    self.channels.get_g(g_xyz)
                };

                let b = if self.channel_configs[2].seamless {
                    sample_seamless(&self.channels, b_channel, b_xyz, Channels::get_b)
                } else {
                    self.channels.get_b(b_xyz)
                };

                let a = if self.channel_configs[3].seamless {
                    sample_seamless(&self.channels, a_channel, a_xyz, Channels::get_a)
                } else {
                    self.channels.get_a(a_xyz)
                };

                let [r, g, b, a] = match self.channel_swizzles {
                    ChannelSwizzles::Rgba => [r, g, b, a],
                    ChannelSwizzles::Bgra => [b, g, r, a],
                };

                // Calculate base index in data array for this pixel
                let base_index = index * C::channel_count();

                match C::channel_count() {
                    1 => {
                        data[base_index] = normalize_noise_to_u8(r);
                    }
                    2 => {
                        data[base_index] = normalize_noise_to_u8(r);
                        data[base_index + 1] = normalize_noise_to_u8(g);
                    }
                    3 => {
                        data[base_index] = normalize_noise_to_u8(r);
                        data[base_index + 1] = normalize_noise_to_u8(g);
                        data[base_index + 2] = normalize_noise_to_u8(b);
                    }
                    4 => {
                        data[base_index] = normalize_noise_to_u8(r);
                        data[base_index + 1] = normalize_noise_to_u8(g);
                        data[base_index + 2] = normalize_noise_to_u8(b);
                        data[base_index + 3] = normalize_noise_to_u8(a);
                    }
                    _ => unreachable!(),
                }
            }
        }

        data
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelConfig {
    bounds: ChannelBounds,
    seamless: bool,
}

impl ChannelConfig {
    fn new(bounds: ChannelBounds, seamless: bool) -> Self {
        Self { bounds, seamless }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelBounds {
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub z: (f64, f64),
}

impl Default for ChannelBounds {
    fn default() -> Self {
        Self {
            x: (0.0, 1.0),
            y: (0.0, 1.0),
            z: (0.0, 1.0),
        }
    }
}

impl ChannelBounds {
    pub fn new(x: (f64, f64), y: (f64, f64), z: (f64, f64)) -> Self {
        Self { x, y, z }
    }

    pub fn splat(v: (f64, f64)) -> Self {
        Self::new(v, v, v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelSwizzles {
    #[default]
    Rgba,
    Bgra,
}

pub trait NoiseChannel<T: NoiseFn<f64, 3>> {
    fn into_noise(self) -> T;

    fn bounds(&self) -> ChannelBounds;

    fn seamless(&self) -> bool;
}

impl<T: NoiseFn<f64, 3>> NoiseChannel<T> for T {
    #[inline]
    fn into_noise(self) -> T {
        self
    }

    #[inline]
    fn bounds(&self) -> ChannelBounds {
        ChannelBounds::default()
    }

    #[inline]
    fn seamless(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct NoiseFnChannel<T> {
    noise: T,
    bounds: ChannelBounds,
    seamless: bool,
}

impl<T> NoiseFnChannel<T> {
    pub fn with_seamless(mut self, seamless: bool) -> Self {
        self.seamless = seamless;
        self
    }
}

impl<T: NoiseFn<f64, 3>> NoiseChannel<T> for NoiseFnChannel<T> {
    #[inline]
    fn into_noise(self) -> T {
        self.noise
    }

    #[inline]
    fn bounds(&self) -> ChannelBounds {
        self.bounds
    }

    #[inline]
    fn seamless(&self) -> bool {
        self.seamless
    }
}

pub trait NoiseChannelEx: Sized {
    fn with_bounds(self, bounds: ChannelBounds) -> NoiseFnChannel<Self>;
}

impl<T: NoiseFn<f64, 3> + Sized> NoiseChannelEx for T {
    fn with_bounds(self, bounds: ChannelBounds) -> NoiseFnChannel<Self> {
        NoiseFnChannel {
            noise: self,
            bounds,
            seamless: false,
        }
    }
}

pub trait Channels {
    fn channel_count() -> usize;

    fn get_r(&self, _point: [f64; 3]) -> f64 {
        0.0
    }

    #[inline(always)]
    fn get_g(&self, _point: [f64; 3]) -> f64 {
        0.0
    }

    #[inline(always)]
    fn get_b(&self, _point: [f64; 3]) -> f64 {
        0.0
    }

    #[inline(always)]
    fn get_a(&self, _point: [f64; 3]) -> f64 {
        0.0
    }
}

impl Channels for () {
    #[inline(always)]
    fn channel_count() -> usize {
        0
    }

    #[inline(always)]
    fn get_r(&self, _: [f64; 3]) -> f64 {
        unreachable!()
    }

    #[inline(always)]
    fn get_g(&self, _: [f64; 3]) -> f64 {
        unreachable!()
    }

    #[inline(always)]
    fn get_b(&self, _: [f64; 3]) -> f64 {
        unreachable!()
    }

    #[inline(always)]
    fn get_a(&self, _: [f64; 3]) -> f64 {
        unreachable!()
    }
}

#[derive(Debug, Clone)]
pub struct ChannelR<R>(R);

impl<R> Channels for ChannelR<R>
where
    R: NoiseFn<f64, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        1
    }

    #[inline(always)]
    fn get_r(&self, point: [f64; 3]) -> f64 {
        self.0.get(point)
    }
}

#[derive(Debug, Clone)]
pub struct ChannelRg<R, G>(R, G);

impl<R, G> Channels for ChannelRg<R, G>
where
    R: NoiseFn<f64, 3>,
    G: NoiseFn<f64, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        2
    }

    #[inline(always)]
    fn get_r(&self, point: [f64; 3]) -> f64 {
        self.0.get(point)
    }

    #[inline(always)]
    fn get_g(&self, point: [f64; 3]) -> f64 {
        self.1.get(point)
    }
}

#[derive(Debug, Clone)]
pub struct ChannelRgb<R, G, B>(R, G, B);

impl<R, G, B> Channels for ChannelRgb<R, G, B>
where
    R: NoiseFn<f64, 3>,
    G: NoiseFn<f64, 3>,
    B: NoiseFn<f64, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        3
    }

    #[inline(always)]
    fn get_r(&self, point: [f64; 3]) -> f64 {
        self.0.get(point)
    }

    #[inline(always)]
    fn get_g(&self, point: [f64; 3]) -> f64 {
        self.1.get(point)
    }

    #[inline(always)]
    fn get_b(&self, point: [f64; 3]) -> f64 {
        self.2.get(point)
    }
}

#[derive(Debug, Clone)]
pub struct ChannelRgba<R, G, B, A>(R, G, B, A);

impl<R, G, B, A> Channels for ChannelRgba<R, G, B, A>
where
    R: NoiseFn<f64, 3>,
    G: NoiseFn<f64, 3>,
    B: NoiseFn<f64, 3>,
    A: NoiseFn<f64, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        4
    }

    #[inline(always)]
    fn get_r(&self, point: [f64; 3]) -> f64 {
        self.0.get(point)
    }

    #[inline(always)]
    fn get_g(&self, point: [f64; 3]) -> f64 {
        self.1.get(point)
    }

    #[inline(always)]
    fn get_b(&self, point: [f64; 3]) -> f64 {
        self.2.get(point)
    }

    #[inline(always)]
    fn get_a(&self, point: [f64; 3]) -> f64 {
        self.3.get(point)
    }
}

fn normalize_noise_to_u8(noise: f64) -> u8 {
    ((noise * 0.5 + 0.5) * 255.0) as u8
}
