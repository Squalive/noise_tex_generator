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
    size: [u32; 3],
    seamless: bool,
    channel_bounds: [ChannelBounds; 4],
    channel_swizzles: ChannelSwizzles,
    channels: C,
}

impl Default for NoiseTextureDescriptor<()> {
    fn default() -> Self {
        Self {
            size: [1, 1, 1],
            seamless: false,
            channel_bounds: Default::default(),
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

    pub fn with_seamless(mut self, seamless: bool) -> Self {
        self.seamless = seamless;
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
            seamless: self.seamless,
            channel_bounds: [
                r.bounds(),
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
            seamless: self.seamless,
            channel_bounds: [
                r.bounds(),
                g.bounds(),
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
            seamless: self.seamless,
            channel_bounds: [r.bounds(), g.bounds(), b.bounds(), Default::default()],
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
            seamless: self.seamless,
            channel_bounds: [r.bounds(), g.bounds(), b.bounds(), a.bounds()],
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

        let [width, height, depth] = self.size;

        let mut data = vec![0u8; (width * height * depth) as usize * C::channel_count()];

        if let Some(len) = data.len().checked_div(C::channel_count()) {
            let seamless = self.seamless;

            let r_channel = PerChannelValues::new(width, height, depth, self.channel_bounds[0]);
            let g_channel = PerChannelValues::new(width, height, depth, self.channel_bounds[1]);
            let b_channel = PerChannelValues::new(width, height, depth, self.channel_bounds[2]);
            let a_channel = PerChannelValues::new(width, height, depth, self.channel_bounds[3]);

            let wh = width * height;
            for index in 0..len {
                let z = index as u32 / wh;
                let slice_index = index as u32 % wh;
                let y = slice_index / width;
                let x = slice_index % width;

                let r_xyz = r_channel.get_xyz(x, y, z);
                let g_xyz = r_channel.get_xyz(x, y, z);
                let b_xyz = r_channel.get_xyz(x, y, z);
                let a_xyz = r_channel.get_xyz(x, y, z);

                let [r, g, b, a] = if seamless {
                    let values = [
                        r_channel.get_seamless_values(r_xyz[0], r_xyz[1], r_xyz[2]),
                        g_channel.get_seamless_values(g_xyz[0], g_xyz[1], g_xyz[2]),
                        b_channel.get_seamless_values(b_xyz[0], b_xyz[1], b_xyz[2]),
                        a_channel.get_seamless_values(a_xyz[0], a_xyz[1], a_xyz[2]),
                    ];

                    // Batch sample all 8 points using array
                    let samples: [[f64; 4]; 8] = [
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[0],
                            values[1].sample_coords[0],
                            values[2].sample_coords[0],
                            values[3].sample_coords[0],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[1],
                            values[1].sample_coords[1],
                            values[2].sample_coords[1],
                            values[3].sample_coords[1],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[2],
                            values[1].sample_coords[2],
                            values[2].sample_coords[2],
                            values[3].sample_coords[2],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[3],
                            values[1].sample_coords[3],
                            values[2].sample_coords[3],
                            values[3].sample_coords[3],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[4],
                            values[1].sample_coords[4],
                            values[2].sample_coords[4],
                            values[3].sample_coords[4],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[5],
                            values[1].sample_coords[5],
                            values[2].sample_coords[5],
                            values[3].sample_coords[5],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[6],
                            values[1].sample_coords[6],
                            values[2].sample_coords[6],
                            values[3].sample_coords[6],
                        )),
                        self.channels.get(SamplePoint::new(
                            values[0].sample_coords[7],
                            values[1].sample_coords[7],
                            values[2].sample_coords[7],
                            values[3].sample_coords[7],
                        )),
                    ];

                    // Trilinear interpolation for each channel
                    let mut result = [0.0; 4];

                    // Process each channel independently
                    for channel in 0..4 {
                        // Extract channel values - all stack allocated
                        let c000 = samples[0][channel];
                        let c100 = samples[1][channel];
                        let c010 = samples[2][channel];
                        let c110 = samples[3][channel];
                        let c001 = samples[4][channel];
                        let c101 = samples[5][channel];
                        let c011 = samples[6][channel];
                        let c111 = samples[7][channel];

                        // X interpolation
                        let c00 = lerp(c000, c100, values[channel].x_blend);
                        let c01 = lerp(c010, c110, values[channel].x_blend);
                        let c10 = lerp(c001, c101, values[channel].x_blend);
                        let c11 = lerp(c011, c111, values[channel].x_blend);

                        // Y interpolation
                        let c0 = lerp(c00, c01, values[channel].y_blend);
                        let c1 = lerp(c10, c11, values[channel].y_blend);

                        // Z interpolation
                        result[channel] = lerp(c0, c1, values[channel].z_blend);
                    }

                    result
                } else {
                    self.channels
                        .get(SamplePoint::new(r_xyz, g_xyz, b_xyz, a_xyz))
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

#[derive(Clone, Copy)]
pub struct SamplePoint {
    pub r: [f64; 3],
    pub g: [f64; 3],
    pub b: [f64; 3],
    pub a: [f64; 3],
}

impl SamplePoint {
    fn new(r: [f64; 3], g: [f64; 3], b: [f64; 3], a: [f64; 3]) -> Self {
        Self { r, g, b, a }
    }
}

pub trait NoiseChannel<T: NoiseFn<f64, 3>> {
    fn into_noise(self) -> T;

    fn bounds(&self) -> ChannelBounds;
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
}

#[derive(Debug, Clone)]
pub struct Bounded<T> {
    noise: T,
    bounds: ChannelBounds,
}

impl<T: NoiseFn<f64, 3>> NoiseChannel<T> for Bounded<T> {
    #[inline]
    fn into_noise(self) -> T {
        self.noise
    }

    #[inline]
    fn bounds(&self) -> ChannelBounds {
        self.bounds
    }
}

pub trait NoiseChannelEx: Sized {
    fn with_bounds(self, bounds: ChannelBounds) -> Bounded<Self>;
}

impl<T: NoiseFn<f64, 3> + Sized> NoiseChannelEx for T {
    fn with_bounds(self, bounds: ChannelBounds) -> Bounded<Self> {
        Bounded {
            noise: self,
            bounds,
        }
    }
}

pub trait Channels {
    fn channel_count() -> usize;

    fn get(&self, point: SamplePoint) -> [f64; 4];
}

impl Channels for () {
    #[inline(always)]
    fn channel_count() -> usize {
        0
    }

    #[inline(always)]
    fn get(&self, _: SamplePoint) -> [f64; 4] {
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
    fn get(&self, point: SamplePoint) -> [f64; 4] {
        let Self(r) = self;
        [r.get(point.r), 0.0, 0.0, 0.0]
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
    fn get(&self, point: SamplePoint) -> [f64; 4] {
        let Self(r, g) = self;
        [r.get(point.r), g.get(point.g), 0.0, 0.0]
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
    fn get(&self, point: SamplePoint) -> [f64; 4] {
        let Self(r, g, b) = self;
        [r.get(point.r), g.get(point.g), b.get(point.b), 0.0]
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
    fn get(&self, point: SamplePoint) -> [f64; 4] {
        let Self(r, g, b, a) = self;
        [
            r.get(point.r),
            g.get(point.g),
            b.get(point.b),
            a.get(point.a),
        ]
    }
}

fn normalize_noise_to_u8(noise: f64) -> u8 {
    ((noise * 0.5 + 0.5) * 255.0) as u8
}
