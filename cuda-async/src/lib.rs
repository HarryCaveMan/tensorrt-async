

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

    #[test]

    fn test_concurrent_roundtrips() {
        let res: CuResult<()> =  Runtime::new().unwrap().block_on(async {
            cu_init().unwrap();
            println!("Initializing CUDA");
            let device = CuDevice::new(0)?;
            println!("Creating context");
            let _ctx = CuContext::new(&device)?;
            println!("Creating stream");
            let stream = CuStream::new()?;
            let max_shape = [16, 128, 512];
            let buffer_size: usize = max_shape.iter().product::<usize>() * size_of::<f32>();
            let arr1_shape = [4, 32, 512];
            let arr2_shape = [8, 64, 512];
            let arr3_shape = [16, 128, 512];
            let arr4_shape = [4, 64, 512];
            let arr5_shape = [8, 128, 512];
            let arr6_shape = [16, 64, 512];
            let arr7_shape = [4, 32, 256];
            let arr1 = ArrayD::<f32>::ones(IxDyn(&arr1_shape));
            let arr2 = ArrayD::<f32>::zeros(IxDyn(&arr2_shape));
            let arr3 = ArrayD::<f32>::ones(IxDyn(&arr3_shape));
            let arr4 = ArrayD::<f32>::zeros(IxDyn(&arr4_shape));
            let arr5 = ArrayD::<f32>::ones(IxDyn(&arr5_shape));
            let arr6 = ArrayD::<f32>::zeros(IxDyn(&arr6_shape));
            let arr7 = ArrayD::<f32>::ones(IxDyn(&arr7_shape));
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
            assert_eq!(arr1, arr1_after_roundtrip);
            assert_eq!(arr2, arr2_after_roundtrip);
            assert_eq!(arr3, arr3_after_roundtrip);
            assert_eq!(arr4, arr4_after_roundtrip);
            assert_eq!(arr5, arr5_after_roundtrip);
            assert_eq!(arr6, arr6_after_roundtrip);
            Ok(())
        });
        assert_eq!(res, Ok(()));
    }

    async fn test_roundtrip(start: &Instant ,id: usize, example_data: &ArrayD<f32>, buffer_size: usize, stream: &CuStream) -> CuResult<ArrayD<f32>>{
        
        println!("ID {} Start: {}",id,start.elapsed().as_millis());
        let example_data_size: usize = example_data.len() * size_of::<f32>();
        println!("Creating buffers");
        let mut io_buffers: IOBuffers = IOBuffers::new::<f32>(buffer_size, buffer_size, &stream)?;
        let data_after_roundtrip: ArrayD<f32> = io_buffers.with_guard_mut(
            example_data_size,
            |io_buf| async {
                println!("ID {} Taints: {:?}", id, io_buf.taints());
                assert_eq!(io_buf.taints(),example_data_size);
                println!("Loading data to input buffer: {}", id);
                io_buf.push::<f32>(&example_data).await?;
                println!("Moving data to output device buffer: {}", id);
                io_buf.itod().await?;
                println!("Dumping data from output buffer: {}", id);      
                io_buf.pull::<f32>().await
            }
        ).await?;
        println!("ID {} Taints: {:?}", id, io_buffers.taints());
        println!("ID {} Complete: {}", id, (start.elapsed().as_millis()));
        assert_eq!(io_buffers.taints(),0);
        Ok(data_after_roundtrip)
    }
}