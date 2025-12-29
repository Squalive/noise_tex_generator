#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec;
use alloc::vec::Vec;
use noise::NoiseFn;

type NoiseT = f64;

#[derive(Debug, Clone)]
pub struct NoiseTextureDescriptor<C: Channels> {
    extent: [u32; 3],
    scale: f64,
    channels: C,
}

impl Default for NoiseTextureDescriptor<()> {
    fn default() -> Self {
        Self {
            extent: [1, 1, 1],
            scale: 1.0,
            channels: (),
        }
    }
}

impl<C: Channels> NoiseTextureDescriptor<C> {
    pub fn with_extent(mut self, extent: [u32; 3]) -> Self {
        self.extent = extent;
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    pub fn to_texture(&self) -> Vec<u8> {
        let [width, height, depth] = self.extent;

        let mut data = vec![0u8; (width * height * depth) as usize * C::channel_count()];

        if let Some(len) = data.len().checked_div(C::channel_count()) {
            let inv_width_scale = 1.0 / width as f64 * self.scale;
            let inv_height_scale = 1.0 / height as f64 * self.scale;
            let inv_depth_scale = 1.0 / depth as f64 * self.scale;

            let wh = width * height;
            for index in 0..len {
                let z = index as u32 / wh;
                let slice_index = index as u32 % wh;
                let y = slice_index / width;
                let x = slice_index % width;

                let [r, g, b, a] = self.channels.get([
                    x as f64 * inv_width_scale,
                    y as f64 * inv_height_scale,
                    z as f64 * inv_depth_scale,
                ]);

                // Calculate base index in data array for this pixel
                let base_index = index * C::channel_count();

                match C::channel_count() {
                    1 => {
                        data[base_index] = r;
                    }
                    2 => {
                        data[base_index] = r;
                        data[base_index + 1] = g;
                    }
                    3 => {
                        data[base_index] = r;
                        data[base_index + 1] = g;
                        data[base_index + 2] = b;
                    }
                    4 => {
                        data[base_index] = r;
                        data[base_index + 1] = g;
                        data[base_index + 2] = b;
                        data[base_index + 3] = a;
                    }
                    _ => unreachable!(),
                }
            }
        }

        data
    }
}

impl NoiseTextureDescriptor<()> {
    pub fn with_r<T: NoiseFn<NoiseT, 3>>(self, noise: T) -> NoiseTextureDescriptor<ChannelR<T>> {
        let Self { extent, scale, .. } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelR(noise),
        }
    }
}

impl<R> NoiseTextureDescriptor<ChannelR<R>>
where
    R: NoiseFn<NoiseT, 3>,
{
    pub fn with_r<T: NoiseFn<NoiseT, 3>>(self, noise: T) -> NoiseTextureDescriptor<ChannelR<T>> {
        let Self { extent, scale, .. } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelR(noise),
        }
    }

    pub fn with_g<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRg<R, T>> {
        let Self {
            extent,
            scale,
            channels: ChannelR(r),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRg(r, noise),
        }
    }
}

impl<R, G> NoiseTextureDescriptor<ChannelRg<R, G>>
where
    R: NoiseFn<NoiseT, 3>,
    G: NoiseFn<NoiseT, 3>,
{
    pub fn with_r<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRg<T, G>> {
        let Self {
            extent,
            scale,
            channels: ChannelRg(_, g),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRg(noise, g),
        }
    }

    pub fn with_g<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRg<R, T>> {
        let Self {
            extent,
            scale,
            channels: ChannelRg(r, _),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRg(r, noise),
        }
    }

    pub fn with_b<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRgb<R, G, T>> {
        let Self {
            extent,
            scale,
            channels: ChannelRg(r, g),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRgb(r, g, noise),
        }
    }
}

impl<R, G, B> NoiseTextureDescriptor<ChannelRgb<R, G, B>>
where
    R: NoiseFn<NoiseT, 3>,
    G: NoiseFn<NoiseT, 3>,
    B: NoiseFn<NoiseT, 3>,
{
    pub fn with_r<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRgb<T, G, B>> {
        let Self {
            extent,
            scale,
            channels: ChannelRgb(_, g, b),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRgb(noise, g, b),
        }
    }

    pub fn with_g<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRgb<R, T, B>> {
        let Self {
            extent,
            scale,
            channels: ChannelRgb(r, _, b),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRgb(r, noise, b),
        }
    }

    pub fn with_b<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRgb<R, G, T>> {
        let Self {
            extent,
            scale,
            channels: ChannelRgb(r, g, _),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRgb(r, g, noise),
        }
    }

    pub fn with_a<T: NoiseFn<NoiseT, 3>>(
        self,
        noise: T,
    ) -> NoiseTextureDescriptor<ChannelRgba<R, G, B, T>> {
        let Self {
            extent,
            scale,
            channels: ChannelRgb(r, g, b),
        } = self;
        NoiseTextureDescriptor {
            extent,
            scale,
            channels: ChannelRgba(r, g, b, noise),
        }
    }
}

pub trait Channels {
    fn channel_count() -> usize;

    fn get(&self, point: [NoiseT; 3]) -> [u8; 4];
}

impl Channels for () {
    #[inline(always)]
    fn channel_count() -> usize {
        0
    }

    #[inline(always)]
    fn get(&self, _: [NoiseT; 3]) -> [u8; 4] {
        unreachable!()
    }
}

#[derive(Debug, Clone)]
pub struct ChannelR<R>(R);

impl<R> Channels for ChannelR<R>
where
    R: NoiseFn<NoiseT, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        1
    }

    #[inline(always)]
    fn get(&self, point: [NoiseT; 3]) -> [u8; 4] {
        let Self(r) = self;
        let r = normalize_noise_to_u8(r.get(point));
        [r, 0, 0, 0]
    }
}

#[derive(Debug, Clone)]
pub struct ChannelRg<R, G>(R, G);

impl<R, G> Channels for ChannelRg<R, G>
where
    R: NoiseFn<NoiseT, 3>,
    G: NoiseFn<NoiseT, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        2
    }

    #[inline(always)]
    fn get(&self, point: [NoiseT; 3]) -> [u8; 4] {
        let Self(r, g) = self;
        let r = normalize_noise_to_u8(r.get(point));
        let g = normalize_noise_to_u8(g.get(point));
        [r, g, 0, 0]
    }
}

#[derive(Debug, Clone)]
pub struct ChannelRgb<R, G, B>(R, G, B);

impl<R, G, B> Channels for ChannelRgb<R, G, B>
where
    R: NoiseFn<NoiseT, 3>,
    G: NoiseFn<NoiseT, 3>,
    B: NoiseFn<NoiseT, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        3
    }

    #[inline(always)]
    fn get(&self, point: [NoiseT; 3]) -> [u8; 4] {
        let Self(r, g, b) = self;
        let r = normalize_noise_to_u8(r.get(point));
        let g = normalize_noise_to_u8(g.get(point));
        let b = normalize_noise_to_u8(b.get(point));
        [r, g, b, 0]
    }
}

#[derive(Debug, Clone)]
pub struct ChannelRgba<R, G, B, A>(R, G, B, A);

impl<R, G, B, A> Channels for ChannelRgba<R, G, B, A>
where
    R: NoiseFn<NoiseT, 3>,
    G: NoiseFn<NoiseT, 3>,
    B: NoiseFn<NoiseT, 3>,
    A: NoiseFn<NoiseT, 3>,
{
    #[inline(always)]
    fn channel_count() -> usize {
        4
    }

    #[inline(always)]
    fn get(&self, point: [NoiseT; 3]) -> [u8; 4] {
        let Self(r, g, b, a) = self;
        let r = normalize_noise_to_u8(r.get(point));
        let g = normalize_noise_to_u8(g.get(point));
        let b = normalize_noise_to_u8(b.get(point));
        let a = normalize_noise_to_u8(a.get(point));
        [r, g, b, a]
    }
}

fn normalize_noise_to_u8(noise: f64) -> u8 {
    ((noise * 0.5 + 0.5) * 255.0) as u8
}
