use cuda::{
    CuStream
};
use trt_api::{
    runtime::{
        Runtime,
        ExecutionContext
    }
};
use trt{
    error::{
        TRTResult,
        TRTError
    }
};
use crate::context;
use std::path::PathBuf; 

struct TRTModel<'runtime> {
    runtime: Runtime,
    context_pool: Vec<ExecutionContext>
}

impl<'runtime> for TRTmodel<'runtime> {
    pub fn new(path: PathBuf, num_contexts: usize, num_streams: usize, num_buffers_per_stream: usize) -> TRTResult<Self> {
        let runtime = Runtime::new()?;
        let engine = runtime.deserialize_engine(path)?;
        let context_pool = context::TRTExecPool(
            &'runtime engine,
            num_contexts,
            num_streams,
            num_buffers_per_stream
        )
    }

    pub async fn infer<I,O>(&self, inputs: &HashMap<&str, Array<I>) -> TRTResult<O> {
        let result = self.context_pool.with_context(|context,stream,buf| async {
            context.execute_v3(inputs).await?
        }).await?;
        Ok(result)
    }
}