use cuda::{
    CuEvent,
    CuResult,
    stream::CuStreamPool,
    atomics::{
        Guard,
        Taint
    }
    macros::wrap_async;
};
use trt_api::{
    runtime::{
        OptProfileSelector,
        TensorIOMode,
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
    collections::HashMap,
    sync::{
        Arc,
        atomic::{
            AtomicUsize,
            Ordering::SeqCst
        }
    }
};

pub struct AsyncTRTContext<'engine> {
    stream_pool: CuStreamPool<'engine>,
    context: RefCell<ExecutionContext>,
    inputs_ready: CuEvent,
    outputs_ready: CuEvent,
    exec_event: CuEvent,
    guard: Guard,
    taints: Arc<AtomicUsize>
}

impl<'engine> AsyncTRTContext<'engine> {
    pub fn new(engine: &'engine Engine, num_streams: usize, buffers_per_stream: usize) -> TRTResult<Self> {
        let context = match engine.create_execution_context(None) {
            Some(ctx) => Ok(ctx),
            None => Err(TRTError::ContextCreationError)
        }?;
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for tensor_index in 0..engine.get_num_io_tensors() {
            let tensor_name = engine.get_io_tensor_name(tensor_index);            
            let tensor_element_size = engine.get_io_tensor_dtype(tensor_name).get_elem_size();
            
            let tensor_mode = engine.get_io_tensor_mode(tensor_name);
            if tensor_mode.is_input() {
                let tensor_max_shape = engine.get_profile_shape(tensor_name, 0, OptProfileSelector::Max);
                context.set_tensor_shape(tensor_name, tensor_max_shape)
                let buffer_size = tensor_max_shape.iter().product::<usize>() * tensor_element_size;
                inputs.insert(tensor_name, buffer_size);
            } else {
                outputs.insert(tensor_name, 0);
            }
        }
        if !context.infer_shapes() {
            return Err(TRTError::ShapeInferenceError);
        }
        for output_name in outputs.keys() {
            let tensor_max_shape = context.get_tensor_shape(output_name);
            let buffer_size = tensor_max_shape.iter().product::<usize>() * tensor_element_size;
            outputs.insert(output_name, buffer_size);
        }

        let stream_pool = CuStreamPool::new(
            num_streams,
            buffers_per_stream,
            inputs,
            outputs
        )?;
        let this = Self {
            stream_pool: stream_pool,
            context: RefCell::new(context),
            inputs_ready: CuEvent::new()?,
            outputs_ready: CuEvent::new()?,
            exec_event: CuEvent::new()?,
            guard: Guard::new(),
            taints: Arc::new(AtomicUsize::new(0))
        };
        if this.context.set_input_consumed_event(this.inputs_ready) {
            Ok(this)
        }
        else {
            Err(TRTError::ContextCreationError)
        }
    }

    pub async fn execute_v3<'context, I, O>(&self, input_tensors: HashMap<&'context str, Array<I>) -> CuResult<O> 
    where
        I: Clone + 'context,
        O: Clone + 'context
    {
        let taint_size = input_tensors.iter().map(|(name, tensor)| tensor.len()*size_of::<I>()).sum();
        self.context_with_stream_and_buffers(taint_size, move |context, stream, buffers| async move {
            buffers.inputs.push_inputs::<I>(input_tensors).await?;
            for (name,buffer) in buffers.inputs.borrow().iter() {
                context.set_input_shape(name, buffer.shape());
                context.set_tensor_address(name, buffer.device_ptr());
            }
            for (name,buffer) in buffers.outputs.borrow().iter() {
                context.set_tensor_address(name, buffer.device_ptr());
            }
            self.inputs_ready.record(stream)?;          
            if context.enqueue_v3(stream) {
                self.exec_event.record(stream)?;
                let event = &self.exec_event;
                wrap_async!(Ok(()),event,stream).await?;
                Ok(buffers.outputs.borrow().pull_outputs::<O>().await?)
            }
            else {
                Err(CuError::InvalidValue)
            }
        }).await
    }

    pub async fn context_with_stream_and_buffers<'context, F, Fut, R>(&'context self, size: usize, func: F) -> R
    where
        F: FnOnce(&'context mut ExecutionContext, &'context UnsafeIOBuffers<'engine>) -> Fut,
        Fut: Future<Output = R> + 'context,
    {
        let _taint = Taint::new(self.taints.clone(), size);
        self.guard.with_lock(|| async {
            let mut context = self.context.borrow_mut();
            self.stream_pool.with_buffers(size, |stream,buf| func(context,stream,buf)).await
        }).await
    }
}

struct TRTExecPool<'engine> {
    pool: Vec<AsyncTRTContext>
}

impl<'engine> TRTExecPool<'engine> {
    pub fn new(engine: &'engine Engine, num_execs: usize, num_streams: usize, buffers_per_stream: usize) -> TRTResult<Self> {
        let mut pool = Vec::new();
        for _ in 0..num_execs {
            pool.push(AsyncTRTContext::new(engine, num_streams, buffers_per_stream)?);
        }
        Ok(Self {
            pool
        })
    }

    pub async fn with_context<'context, F, Fut, R>(&'context self, size: usize, func: F) -> R
    where
        F: FnOnce(&'context ExecutionContext, &'context CUcontext, &'context UnsafeIOBuffers<'engine>) -> Fut,
        Fut: Future<Output = R> + 'context
    {
       let context = self.pool.iter().min_by_key(|context| context.taints()).expect("Empty context pool not supported!");
         context.context_with_stream_and_buffers(size, |context,stream,buf| func(context,stream,buf)).await
    }
}