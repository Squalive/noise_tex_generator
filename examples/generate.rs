use noise::{Constant, Negate, Worley};
use nt_lib::{ChannelSwizzles, NoiseTextureDescriptor};
use nvtt_rs::{CUDA_SUPPORTED, CompressionOptions, Context, InputFormat, OutputOptions, Surface};

fn main() {
    let width = 128;
    let height = 128;
    let depth = 128;

    let bounds = (0.0, 8.0);
    // output_with_planemap(width, height, bounds);
    output_with_nvtt(width, height, depth, bounds);
}

// fn output_with_planemap(width: u32, height: u32, bounds: (f64, f64)) {
//     let map = PlaneMapBuilder::new(Perlin::default())
//         .set_is_seamless(true)
//         .set_x_bounds(bounds.0, bounds.1)
//         .set_y_bounds(bounds.0, bounds.1)
//         .set_size(width as _, height as _)
//         .build();

//     let mut data = Vec::with_capacity(width as usize * height as usize * 4);
//     for value in map.iter() {
//         data.push(((*value * 0.5 + 0.5) * 255.0) as u8);
//         data.push(0u8);
//         data.push(0u8);
//         data.push(255);
//     }

//     image::ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data)
//         .unwrap()
//         .save("noise.png")
//         .unwrap();
// }

fn output_with_nvtt(width: u32, height: u32, depth: u32, bounds: (f64, f64)) {
    let noise = Negate::new(
        Worley::default()
            .set_return_type(noise::core::worley::ReturnType::Distance)
            .set_frequency(0.5),
    );
    let data = NoiseTextureDescriptor::default()
        .with_size([width, height, depth])
        .with_bounds([bounds; 3])
        .with_seamless(true)
        .with_channel_swizzles(ChannelSwizzles::Bgra)
        .with_rgba(
            noise,
            Constant::new(-1.0),
            Constant::new(-1.0),
            Constant::new(1.0),
        )
        .to_texture();

    let input = InputFormat::Bgra8Ub {
        data: &data,
        unsigned_to_signed: false,
    };
    let image = Surface::image(input, width, height, depth).unwrap();

    let mut context = Context::new();
    if *CUDA_SUPPORTED {
        context.set_cuda_acceleration(true);
        println!("--CUDA ACCELERATED COMPRESSION")
    }

    let mut compression_options = CompressionOptions::default();
    compression_options.set_format(nvtt_rs::Format::Bc7);
    compression_options.set_quality(nvtt_rs::Quality::Normal);

    let mut output_options = OutputOptions::new();
    output_options.set_srgb_flag(false);

    let header = context
        .output_header(&image, 1, &compression_options, &output_options)
        .unwrap();

    let bytes = context
        .compress(&image, &compression_options, &output_options)
        .unwrap();

    let dds_data = [header, bytes].concat();

    std::fs::write("noise.dds", &dds_data).unwrap();
}
