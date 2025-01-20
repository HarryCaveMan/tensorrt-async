use cuda::{
    CuStream,
    io_buffers::IOBufferPool
};
use trt_api::{
    runtime::{
        Engine,
        ExecutionContext
    }
};
use trt{
    error::{
        TRTResult,
        TRTError
    }
};
use std::{
    cell::RefCell,
    collections::HashMap
};

pub struct UnsafeAsyncTRTContext<'engine> {
    stream_pool: Vec<CuStream>,
    buffer_pool: Vec<IOBuffers>,
    context: RefCell<ExecutionContext>
}

impl<'engine> UnsafeAsyncTRTContext<'engine> {
    pub fn new(engine: &'engine Engine, num_streams: usize, num_buffers: usize) -> TRTResult<Self> {
        let context = match engine.create_execution_context() {
            Some(ctx) => Ok(ctx),
            None => Err(TRTError::ContextCreationError)
        }?;
        let mut stream_pool = Vec::new();
        let input_tensors = 
        let mut buffer_pool = IOBufferPool::new::<f32>(num_buffers, HashMap::new(), HashMap::new())?;
        
    }
}

struct TRTExecPool<'engine> {
    pool: Vec<AsyncTRTContext>
}