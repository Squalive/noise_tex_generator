use image::Rgb;
use noise::{Blend, Constant, Fbm, Perlin, PerlinSurflet, Worley, utils::PlaneMapBuilder};
use noise_tex_generator::NoiseTextureDescriptor;

fn main() {
    let width = 512;
    let height = 512;

    let noise = PerlinSurflet::default();

    // let noise = RidgedMulti::default().set_sources(vec![
    //     Worley::default().set_return_type(
    //         noise::core::worley::ReturnType::Distance
    //     );
    //     32
    // ]);

    let data = NoiseTextureDescriptor::default()
        .with_extent([width, height, 1])
        .with_scale(8.0)
        .with_r(noise.clone())
        .with_g(Constant::new(-1.0))
        .with_b(Constant::new(-1.0))
        .to_texture();

    image::ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data)
        .unwrap()
        .save("noise.png")
        .expect("failed to save noise.png");
}
