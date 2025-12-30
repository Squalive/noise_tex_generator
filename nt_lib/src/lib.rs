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
    bounds: [(f64, f64); 3],
    seamless: bool,
    channel_swizzles: ChannelSwizzles,
    channels: C,
}

impl Default for NoiseTextureDescriptor<()> {
    fn default() -> Self {
        Self {
            size: [1, 1, 1],
            bounds: [(0.0, 1.0); 3],
            seamless: false,
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

    pub fn with_bounds(mut self, bounds: [(f64, f64); 3]) -> Self {
        self.bounds = bounds;
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

    pub fn with_r<R>(self, r: R) -> NoiseTextureDescriptor<ChannelR<R>>
    where
        R: NoiseFn<f64, 3>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            bounds: self.bounds,
            seamless: self.seamless,
            channel_swizzles: self.channel_swizzles,
            channels: ChannelR(r),
        }
    }

    pub fn with_rg<R, G>(self, r: R, g: G) -> NoiseTextureDescriptor<ChannelRg<R, G>>
    where
        R: NoiseFn<f64, 3>,
        G: NoiseFn<f64, 3>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            bounds: self.bounds,
            seamless: self.seamless,
            channel_swizzles: self.channel_swizzles,
            channels: ChannelRg(r, g),
        }
    }

    pub fn with_rgb<R, G, B>(self, r: R, g: G, b: B) -> NoiseTextureDescriptor<ChannelRgb<R, G, B>>
    where
        R: NoiseFn<f64, 3>,
        G: NoiseFn<f64, 3>,
        B: NoiseFn<f64, 3>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            bounds: self.bounds,
            seamless: self.seamless,
            channel_swizzles: self.channel_swizzles,
            channels: ChannelRgb(r, g, b),
        }
    }

    pub fn with_rgba<R, G, B, A>(
        self,
        r: R,
        g: G,
        b: B,
        a: A,
    ) -> NoiseTextureDescriptor<ChannelRgba<R, G, B, A>>
    where
        R: NoiseFn<f64, 3>,
        G: NoiseFn<f64, 3>,
        B: NoiseFn<f64, 3>,
        A: NoiseFn<f64, 3>,
    {
        NoiseTextureDescriptor {
            size: self.size,
            bounds: self.bounds,
            seamless: self.seamless,
            channel_swizzles: self.channel_swizzles,
            channels: ChannelRgba(r, g, b, a),
        }
    }

    pub fn to_texture(&self) -> Vec<u8> {
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

            let [x_bounds, y_bounds, z_bounds] = self.bounds;

            let x_extent = x_bounds.1 - x_bounds.0;
            let y_extent = y_bounds.1 - y_bounds.0;
            let z_extent = z_bounds.1 - z_bounds.0;

            let x_step = x_extent / width as f64;
            let y_step = y_extent / height as f64;
            let z_step = z_extent / depth as f64;

            let wh = width * height;
            for index in 0..len {
                let z = index as u32 / wh;
                let slice_index = index as u32 % wh;
                let y = slice_index / width;
                let x = slice_index % width;

                let cur_x = x_bounds.0 + x_step * x as f64;
                let cur_y = y_bounds.0 + y_step * y as f64;
                let cur_z = z_bounds.0 + z_step * z as f64;

                let [r, g, b, a] = if seamless {
                    // Pre-calculate all sample coordinates
                    let sample_coords = [
                        [cur_x, cur_y, cur_z],
                        [cur_x + x_extent, cur_y, cur_z],
                        [cur_x, cur_y + y_extent, cur_z],
                        [cur_x + x_extent, cur_y + y_extent, cur_z],
                        [cur_x, cur_y, cur_z + z_extent],
                        [cur_x + x_extent, cur_y, cur_z + z_extent],
                        [cur_x, cur_y + y_extent, cur_z + z_extent],
                        [cur_x + x_extent, cur_y + y_extent, cur_z + z_extent],
                    ];

                    // Calculate blend factors
                    let x_blend = 1.0 - (cur_x - x_bounds.0) / x_extent;
                    let y_blend = 1.0 - (cur_y - y_bounds.0) / y_extent;
                    let z_blend = 1.0 - (cur_z - z_bounds.0) / z_extent;

                    // Batch sample all 8 points using array
                    let samples: [[f64; 4]; 8] = [
                        self.channels.get(sample_coords[0]),
                        self.channels.get(sample_coords[1]),
                        self.channels.get(sample_coords[2]),
                        self.channels.get(sample_coords[3]),
                        self.channels.get(sample_coords[4]),
                        self.channels.get(sample_coords[5]),
                        self.channels.get(sample_coords[6]),
                        self.channels.get(sample_coords[7]),
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
                        let c00 = lerp(c000, c100, x_blend);
                        let c01 = lerp(c010, c110, x_blend);
                        let c10 = lerp(c001, c101, x_blend);
                        let c11 = lerp(c011, c111, x_blend);

                        // Y interpolation
                        let c0 = lerp(c00, c01, y_blend);
                        let c1 = lerp(c10, c11, y_blend);

                        // Z interpolation
                        result[channel] = lerp(c0, c1, z_blend);
                    }

                    result
                } else {
                    self.channels.get([cur_x, cur_y, cur_z])
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelSwizzles {
    #[default]
    Rgba,
    Bgra,
}

pub trait Channels {
    fn channel_count() -> usize;

    fn get(&self, point: [f64; 3]) -> [f64; 4];
}

impl Channels for () {
    #[inline(always)]
    fn channel_count() -> usize {
        0
    }

    #[inline(always)]
    fn get(&self, _: [f64; 3]) -> [f64; 4] {
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
    fn get(&self, point: [f64; 3]) -> [f64; 4] {
        let Self(r) = self;
        [r.get(point), 0.0, 0.0, 0.0]
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
    fn get(&self, point: [f64; 3]) -> [f64; 4] {
        let Self(r, g) = self;
        [r.get(point), g.get(point), 0.0, 0.0]
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
    fn get(&self, point: [f64; 3]) -> [f64; 4] {
        let Self(r, g, b) = self;
        [r.get(point), g.get(point), b.get(point), 0.0]
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
    fn get(&self, point: [f64; 3]) -> [f64; 4] {
        let Self(r, g, b, a) = self;
        [r.get(point), g.get(point), b.get(point), a.get(point)]
    }
}

fn normalize_noise_to_u8(noise: f64) -> u8 {
    ((noise * 0.5 + 0.5) * 255.0) as u8
}
