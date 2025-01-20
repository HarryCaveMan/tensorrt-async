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

struct TRTModel {
    runtime: Runtime,
    context_pool: Vec<ExecutionContext>
}

impl TRTmodel {
    pub fn new(num_contexts: usize,num_streams) -> TRTResult<Self> {
        let runtime = Runtime::new()?;
        let context_pool = Vec<ExecutionContext>::new();
        for _ in 0..num_contexts {
            let context = ExecutionContext::new(&runtime)?;
            context_pool.push(context);
        }
        Ok(Self {
            runtime,
            context_pool
        })
    }

    pub async fn infer(input_tensors: HashMap<String,serde_json::Value>) {

    }
}