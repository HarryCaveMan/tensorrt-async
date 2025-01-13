use cuda::{
    CuStream,
    IOBuffers
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

struct TrtEcxecutionContext {
    stream_pool: Vec<CuStream>,
    buffer_pool: Vec<IOBuffers>,
    context: ExecutionContext
}

struct TRTExecPool {
    pool: Vec<TrtEcxecutionContext>
}

struct IOBufferPool {
    pool: Vec<IOBuffers>
}