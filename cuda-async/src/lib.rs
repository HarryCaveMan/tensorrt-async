

#[macro_use]
extern crate enum_primitive;

extern crate cuda_rs_sys as cu_abi;

#[macro_use]
extern crate cuda_rs as cu_rs;

#[macro_use]
mod macros;

pub (crate) mod future;


// public interface
pub mod memory;
pub mod io_buffers;
pub mod utils;
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
        io_buffers::IOBuffers,
        CuResult,
        CuError,
        CuStream
    };
    use tokio::runtime::Runtime;
    use ndarray::{ArrayD, IxDyn, s};

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
            let buffer_size: usize = max_shape.iter().product::<usize>() * size_of::<f32>();
            let arr_shape = [4, 32, 512];
            let example_data = ArrayD::<f32>::ones(IxDyn(&arr_shape));
            let example_data_size: usize = example_data.len() * size_of::<f32>();
            println!("Creating buffers");
            let mut io_buffers: IOBuffers = IOBuffers::new::<f32>(buffer_size, buffer_size, &stream).unwrap();
            let data_after_roundtrip: ArrayD<f32> = io_buffers.with_guard_mut(
                example_data_size,
                |io_buf| async {
                    println!("Taints: {:?}", io_buf.taints());
                    assert_eq!(io_buf.taints(),example_data_size);
                    println!("Data before trip {:?}", example_data.slice(s![..3, ..3, ..3]));
                    println!("Loading data to input buffer");
                    io_buf.push::<f32>(&example_data).await.unwrap();
                    println!("Moving data to output device buffer");
                    io_buf.itod().await.unwrap();
                    println!("Dumping data from output buffer");      
                    io_buf.pull::<f32>().await
                }
            ).await;
            println!("Taints: {:?}", io_buffers.taints());
            assert_eq!(io_buffers.taints(),0);
            println!("Data after trip {:?}", data_after_roundtrip.slice(s![..3, ..3, ..3]));
            assert_eq!(example_data, data_after_roundtrip);
        });
    }
}