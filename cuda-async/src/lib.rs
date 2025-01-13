

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
    use tokio::{
        try_join,
        runtime::Runtime
    };
    use ndarray::{ArrayD, IxDyn, s};
    use std::time::Instant;
    use std::collections::HashMap;
    use std::sync::Once;
    use std::pin::Pin;

    static CUDA_INIT: Once = Once::new();
    
    fn init_cuda() {
        CUDA_INIT.call_once(|| {
            unsafe {
                cu_init().expect("Failed to initialize CUDA");
            }
        });
    }
    #[tokio::test]
    async fn concurrency_tests<'tests>() {
        test_concurrent_roundtrips().await.unwrap();
    }
    #[tokio::test]
    async fn guard_tests<'tests>() {
        test_concurrent_guards().await.unwrap();
    }
    
    // Ensure concurrency in rut with multiple buffers on the same stream (serialized on accelerator)
    async fn test_concurrent_roundtrips<'tests>() -> CuResult<()> {
        init_cuda();
        let device = CuDevice::new(0)?;
        let _ctx = CuContext::new(&device)?;
        let stream = CuStream::new()?;
        let max_shape = [16, 128, 512];
        let buffer_size: usize = max_shape.iter().product::<usize>() * size_of::<f32>();
        // I don't care that this is not pretty or DRY, it's a test. Submit a PR if it bothers you.
        let arr1_shape = [4, 32, 512];
        let arr2_shape = [8, 64, 512];
        let arr3_shape = [16, 128, 512];
        let arr4_shape = [4, 64, 512];
        let arr5_shape = [8, 128, 512];
        let arr6_shape = [16, 64, 512];
        let arr7_shape = [4, 32, 256];
        let arr1 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr1_shape))));
        let arr2 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr2_shape))));
        let arr3 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr3_shape))));
        let arr4 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr4_shape))));
        let arr5 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr5_shape))));
        let arr6 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr6_shape))));
        let arr7 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr7_shape))));
        let start = Instant::now();
        let arr1_fut = test_roundtrip(&start,1,&arr1,buffer_size,&stream);
        let arr2_fut = test_roundtrip(&start,2,&arr2,buffer_size,&stream);
        let arr3_fut = test_roundtrip(&start,3,&arr3,buffer_size,&stream);
        let arr4_fut = test_roundtrip(&start,4,&arr4,buffer_size,&stream);
        let arr5_fut = test_roundtrip(&start,5,&arr5,buffer_size,&stream);
        let arr6_fut = test_roundtrip(&start,6,&arr6,buffer_size,&stream);
        let arr7_fut = test_roundtrip(&start,7,&arr7,buffer_size,&stream);
        let (arr1_after_roundtrip, arr2_after_roundtrip, arr3_after_roundtrip,
            arr4_after_roundtrip, arr5_after_roundtrip, arr6_after_roundtrip,
            arr7_after_roundtrip) = try_join!(
                arr1_fut, arr2_fut, arr3_fut, 
                arr4_fut, arr5_fut, arr6_fut, arr7_fut
            )?;
        println!("arr1 after trip {:?}", arr1_after_roundtrip.slice(s![..1, ..1, ..1]));
        println!("arr2 after trip {:?}", arr2_after_roundtrip.slice(s![..1, ..1, ..1]));
        println!("arr3 after trip {:?}", arr3_after_roundtrip.slice(s![..1, ..1, ..1]));
        println!("arr4 after trip {:?}", arr4_after_roundtrip.slice(s![..1, ..1, ..1]));
        println!("arr5 after trip {:?}", arr5_after_roundtrip.slice(s![..1, ..1, ..1]));
        println!("arr6 after trip {:?}", arr6_after_roundtrip.slice(s![..1, ..1, ..1]));
        println!("arr7 after trip {:?}", arr7_after_roundtrip.slice(s![..1, ..1, ..1]));
        assert_eq!(*arr3, arr3_after_roundtrip);
        assert_eq!(*arr4, arr4_after_roundtrip);
        assert_eq!(*arr5, arr5_after_roundtrip);
        assert_eq!(*arr6, arr6_after_roundtrip);
        Ok(())
    }

    // ensure the buffer guard can be held concurrently (multiple threads awaiting the same guard)
    async fn test_concurrent_guards<'tests>() -> CuResult<()> {
        init_cuda();
        let device = CuDevice::new(0)?;
        let _ctx = CuContext::new(&device)?;
        let stream = CuStream::new()?;
        let test_input_max_shape = [16, 512];
        let test_output_max_shape = [16, 128, 512];
        let input_buffer_size: usize = test_input_max_shape.iter().product::<usize>() * size_of::<f32>();
        let output_buffer_size: usize = test_output_max_shape.iter().product::<usize>() * size_of::<f32>();
        let arr1_shape = [4, 68];
        let arr2_shape = [8, 39];
        let arr1 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr1_shape))));
        let arr2 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr2_shape))));
        let start = Instant::now();
        let io_buffers = IOBuffers::new::<f32>(
            HashMap::from_iter(
                vec![
                    ("test_input",input_buffer_size),
                ]
            ),
            HashMap::from_iter(
                vec![
                    ("test_output",output_buffer_size),
                ]
            ),
            &stream
        )?;
        let arr1_fut = io_buffers.with_guard(
            arr1.len() * size_of::<f32>(),
            |io_buf| async {
                println!("ID 1 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr1).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("ID 1 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr2_fut = io_buffers.with_guard(
            arr2.len() * size_of::<f32>(),
            |io_buf| async {
                println!("ID 2 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr2).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("ID 2 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let (arr1_after_roundtrip, arr2_after_roundtrip): (ArrayD<f32>, ArrayD<f32>) = try_join!(arr1_fut, arr2_fut)?;
        assert_eq!(*arr1, arr1_after_roundtrip);
        assert_eq!(*arr2, arr2_after_roundtrip);
        Ok(())
    }

    async fn test_roundtrip<'tests>(
        start: &Instant,
        id: usize,
        example_data: &ArrayD<f32>,
        buffer_size: usize,
        stream: &CuStream
    ) -> CuResult<ArrayD<f32>> {        
        println!("ID {} Start: {}",id,start.elapsed().as_millis());
        let example_data_size: usize = example_data.len() * size_of::<f32>();
        println!("Creating buffers");
        let io_buffers = IOBuffers::new::<f32>(
            HashMap::from_iter(vec![("test_input",buffer_size)]),
            HashMap::from_iter(vec![("test_output",buffer_size)]),
            &stream
        )?;
        let data_after_roundtrip: ArrayD<f32> = io_buffers.with_guard(
            example_data_size,
            |io_buf| async {
                println!("ID {} Taints: {:?}", id, io_buf.taints());
                assert_eq!(io_buf.taints(),example_data_size);
                println!("Loading data to input buffer: {}", id);
                io_buf.push::<f32>("test_input",example_data).await?;
                println!("Moving data to output device buffer: {}", id);
                io_buf.itod("test_input","test_output").await?;
                println!("Dumping data from output buffer: {}", id);      
                io_buf.pull::<f32>("test_output").await
            }
        ).await?;
        println!("ID {} Taints: {:?}", id, io_buffers.taints());
        println!("ID {} Complete: {}", id, (start.elapsed().as_millis()));
        assert_eq!(io_buffers.taints(),0);
        Ok(data_after_roundtrip)
    }
}