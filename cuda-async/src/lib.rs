

#[macro_use]
extern crate enum_primitive;

extern crate cuda_rs_sys as cu_api;

#[macro_use]
extern crate cuda_rs as cu_rs;

#[macro_use]
mod macros;

pub (crate) mod future;


// public interface
pub mod memory;
pub mod taint;
pub use cu_rs::{
    init as cu_init,
    context::CuContext,
    device::CuDevice,
    error::{
        CuResult,
        CuError
    },
    stream::CuStream,
    event::CuEvent
};
pub use future::CuEventFuture;

// 
#[cfg(test)]
mod tests {
    use super::{
        cu_init,
        CuContext,
        CuDevice,
        memory::HostDeviceMem,
        CuResult,
        CuError,
        CuStream
    };
    use tokio::runtime::Runtime;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_memory_roundtrip() {
        Runtime::new().unwrap().block_on(async {
            cu_init().unwrap();
            println!("Initializing CUDA");
            let device = CuDevice::new(0).unwrap();
            println!("Creating context");
            let _ctx = CuContext::new(&device).unwrap();
            println!("Creating stream");
            let stream = CuStream::new().unwrap();
            let max_shape = [16, 128, 512];
            let arr_shape = [4, 32, 512];
            let example_data = ArrayD::<f32>::ones(IxDyn(&max_shape));
            println!("Creating buffers");
            let input_buffer = HostDeviceMem::new::<f32>(&max_shape, &stream).unwrap();
            let output_buffer = HostDeviceMem::new::<f32>(&max_shape, &stream).unwrap();
            let data_after_roundtrip: ArrayD<f32> = input_buffer.with_guard(|| async {
                println!("Data before trip {:?}", example_data);
                println!("Loading data to input buffer");
                input_buffer.load_ndarray::<f32>(&example_data);
                println!("Moving data to input device buffer");
                input_buffer.htod().await;
                output_buffer.with_guard(|| async {
                    println!("Moving data to output device buffer");
                    input_buffer.move_on_device(&output_buffer).await;
                    println!("Moving data to output host buffer");
                    output_buffer.dtoh().await;
                    println!("Dumping data from output host buffer");
                    output_buffer.dump_ndarray::<f32>(&max_shape)
                }).await
            }).await;
            println!("Data after trip {:?}", data_after_roundtrip);
            assert_eq!(example_data, data_after_roundtrip);
        });
    }
}