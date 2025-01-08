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
    context_pool: Vec<ExecutionContext>,
    stream_pool: Vec<CuStream>
}