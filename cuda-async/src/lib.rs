extern crate enum_primitive;

extern crate cuda_rs_sys as cu_ffi;

extern crate cuda_rs as cu_rs;

#[macro_use]
mod macros;

pub (crate) mod future;


// public interface
pub mod memory;
pub mod stream;
pub mod io_buffers;
pub mod atomics;
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
        io_buffers::{
            IOBuffers,
            IOBufferPool
        },
        stream::CuStreamPool,
        CuResult,
        CuStream
    };
    use tokio::{
        try_join
    };
    use ndarray::{ArrayD, IxDyn};
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
    async fn buffer_guard_tests<'tests>() {
        match test_concurrent_buffer_guards().await {
            Ok(_) => println!("Concurrent buffer guard test passed!"),
            Err(e) => panic!("Concurrent buffer guard test failed: {:?}", e),
        }
    }

    #[tokio::test]
    async fn buffer_pool_tests<'tests>() {
        match test_buffer_pool().await {
            Ok(_) => println!("Buffer pool test passed!"),
            Err(e) => panic!("Buffer pool test failed: {:?}", e),
        }
    }

    #[tokio::test]
    async fn stream_pool_tests<'tests>() {
        match test_stream_pool().await {
            Ok(_) => println!("Stream pool test passed!"),
            Err(e) => panic!("Stream pool test failed: {:?}", e),
        }
    }

    // ensure the buffer guard can be held concurrently (multiple threads awaiting the same guard)
    async fn test_concurrent_buffer_guards<'tests>() -> CuResult<()> {
        println!("Starting concurrent buffer guard test");
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
        let io_buffers = IOBuffers::new(
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
            stream.clone()
        )?;
        let arr1_fut = io_buffers.with_guard(
            arr1.len() * size_of::<f32>(),
            |io_buf| async {
                println!("GT ID 1 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr1).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("GT ID 1 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr2_fut = io_buffers.with_guard(
            arr2.len() * size_of::<f32>(),
            |io_buf| async {
                println!("GT ID 2 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr2).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("GT ID 2 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let (arr1_after_roundtrip, arr2_after_roundtrip): (ArrayD<f32>, ArrayD<f32>) = try_join!(arr1_fut, arr2_fut)?;
        assert_eq!(*arr1, arr1_after_roundtrip);
        assert_eq!(*arr2, arr2_after_roundtrip);
        println!("Concurrent buffer guard test passed!");
        Ok(())
    }

    async fn test_buffer_pool<'tests>() -> CuResult<()> {
        println!("Starting buffer pool test");
        init_cuda();
        let device = CuDevice::new(0)?;
        let _ctx = CuContext::new(&device)?;
        let stream = CuStream::new()?;
        let test_input_max_shape = [16, 512];
        let test_output_max_shape = [16, 512,];
        let input_buffer_size: usize = test_input_max_shape.iter().product::<usize>() * size_of::<f32>();
        let output_buffer_size: usize = test_output_max_shape.iter().product::<usize>() * size_of::<f32>();
        let arr1_shape = [4, 68];
        let arr2_shape = [8, 39];
        let arr3_shape = [16, 512];
        let arr4_shape = [16, 128];
        let arr1 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr1_shape))));
        let arr2 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr2_shape))));
        let arr3 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr3_shape))));
        let arr4 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr4_shape))));
        let start = Instant::now();
        let pool = IOBufferPool::new(
            2,
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
            stream.clone()
        )?;
        let arr1_fut = pool.with_buffers(
            arr1.len() * size_of::<f32>(),
            |io_buf| async {
                println!("BP ID 1 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr1).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("BP ID 1 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr2_fut = pool.with_buffers(
            arr2.len() * size_of::<f32>(),
            |io_buf| async {
                println!("BP ID 2 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr2).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("BP ID 2 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr3_fut = pool.with_buffers(
            arr3.len() * size_of::<f32>(),
            |io_buf| async {
                println!("BP ID 3 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr3).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("BP ID 3 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr4_fut = pool.with_buffers(
            arr4.len() * size_of::<f32>(),
            |io_buf| async {
                println!("BP ID 4 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr4).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("BP ID 4 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let (arr1_after_roundtrip, arr2_after_roundtrip, arr3_after_roundtrip, arr4_after_roundtrip): (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>, ArrayD<f32>) = try_join!(arr1_fut, arr2_fut, arr3_fut, arr4_fut)?;
        assert_eq!(*arr1, arr1_after_roundtrip);
        assert_eq!(*arr2, arr2_after_roundtrip);
        assert_eq!(*arr3, arr3_after_roundtrip);
        assert_eq!(*arr4, arr4_after_roundtrip);
        println!("Buffer pool test passed!");
        Ok(())
    }

    async fn test_stream_pool<'tests>() -> CuResult<()> {
        println!("Starting stream pool test");
        init_cuda();
        let device = CuDevice::new(0)?;
        let _ctx = CuContext::new(&device)?;
        let test_input_max_shape = [16, 512];
        let test_output_max_shape = [16, 512,];
        let input_buffer_size: usize = test_input_max_shape.iter().product::<usize>() * size_of::<f32>();
        let output_buffer_size: usize = test_output_max_shape.iter().product::<usize>() * size_of::<f32>();
        let arr1_shape = [4, 68];
        let arr2_shape = [8, 39];
        let arr3_shape = [16, 512];
        let arr4_shape = [16, 128];
        let arr1 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr1_shape))));
        let arr2 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr2_shape))));
        let arr3 = Pin::new(Box::new(ArrayD::<f32>::ones(IxDyn(&arr3_shape))));
        let arr4 = Pin::new(Box::new(ArrayD::<f32>::zeros(IxDyn(&arr4_shape))));
        let start = Instant::now();
        let stream_pool = CuStreamPool::new(
            2,
            1,
            HashMap::from_iter(
                vec![
                    ("test_input",input_buffer_size),
                ]
            ),
            HashMap::from_iter(
                vec![
                    ("test_output",output_buffer_size),
                ]
            )
        )?;
        let arr1_fut = stream_pool.stream_with_buffers(
            arr1.len() * size_of::<f32>(),
            |io_buf| async {
                println!("SP ID 1 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr1).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("SP ID 1 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr2_fut = stream_pool.stream_with_buffers(
            arr2.len() * size_of::<f32>(),
            |io_buf| async {
                println!("SP ID 2 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr2).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("SP ID 2 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr3_fut = stream_pool.stream_with_buffers(
            arr3.len() * size_of::<f32>(),
            |io_buf| async {
                println!("SP ID 3 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr3).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("SP ID 3 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let arr4_fut = stream_pool.stream_with_buffers(
            arr4.len() * size_of::<f32>(),
            |io_buf| async {
                println!("SP ID 4 Start: {}",start.elapsed().as_millis());
                io_buf.push::<f32>("test_input",&arr4).await?;
                io_buf.itod("test_input","test_output").await?;
                println!("SP ID 4 About to release guard: {}",start.elapsed().as_millis());
                io_buf.pull::<f32>("test_output").await
            }
        );
        let (arr1_after_roundtrip, arr2_after_roundtrip, arr3_after_roundtrip, arr4_after_roundtrip): (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>, ArrayD<f32>) = try_join!(arr1_fut, arr2_fut, arr3_fut, arr4_fut)?;
        assert_eq!(*arr1, arr1_after_roundtrip);
        assert_eq!(*arr2, arr2_after_roundtrip);
        assert_eq!(*arr3, arr3_after_roundtrip);
        assert_eq!(*arr4, arr4_after_roundtrip);
        println!("Stream pool test passed!");
        Ok(())
    }
}