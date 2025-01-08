use ndarray::{ArrayD};
use crate::{
    memory::HostDeviceMem,
    utils::{
        Taint,
        Guard
    },
    cu_rs::{
        stream::CuStream,
        event::CuEvent,
        error::{CuResult, CuError}
    }
};
use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{
            AtomicUsize,
            Ordering::SeqCst
        }
    }
};

pub struct IOBuffers<'stream> {
    input: HostDeviceMem<'stream>,
    output: HostDeviceMem<'stream>,
    guard: Guard,
    taints: Arc<AtomicUsize>
}

impl<'stream> IOBuffers<'stream> {
    pub fn new<T>(input_size: usize, output_size: usize, stream: &'stream CuStream) -> CuResult<Self> {
        let input = HostDeviceMem::new::<T>(input_size, stream)?;
        let output = HostDeviceMem::new::<T>(output_size, stream)?;
        Ok(Self {
            input,
            output,
            guard: Guard::new(),
            taints: Arc::new(AtomicUsize::new(0))
        })
    }


    pub async fn with_guard_mut<'buf, F, Fut, R>(&'buf mut self, size: usize, func: F) -> R
    where
        F: FnOnce(&'buf mut Self) -> Fut,
        Fut: Future<Output = R> + 'buf,
    {
        let _taint = Taint::new(self.taints.clone(), size);
        let _guard = self.guard.lock().await;
        func(self).await
    }    

    pub fn input(&self) -> &HostDeviceMem {
        &self.input
    }

    pub fn output(&self) -> &HostDeviceMem {
        &self.output
    }

    pub fn input_mut(&'stream mut self) -> &'stream mut HostDeviceMem<'stream> {
        &mut self.input
    }

    pub fn output_mut(&'stream mut self) -> &'stream mut HostDeviceMem<'stream> {
        &mut self.output
    }

    pub fn taints(&self) -> usize {
        self.taints.load(SeqCst)
    }

    pub async fn push<T>(&mut self, src: &ArrayD<T>) -> CuResult<()>
    where T: Clone {
        self.input.load_ndarray(src)?;
        let res: CuResult<()> = self.input.htod().await;
        self.set_input_shape(&src.shape().to_vec());
        res
    }

    // input device buffer to output device buffer (input to output device -> itod)
    pub async fn itod(&mut self) -> CuResult<()> {
        let res: CuResult<()> = self.input.move_on_device(&mut self.output).await;
        self.set_output_shape(self.input.tensor_shape().clone().as_ref());
        res
    }

    pub async fn pull<T>(&mut self) -> ArrayD<T> 
    where T: Clone + Default {
        self.output.dtoh().await;
        self.output.dump_ndarray()
    }

    pub fn set_input_shape(&mut self, shape: &[usize]) {
        self.input.set_tensor_shape(Vec::<usize>::from(shape).as_ref());
    }

    pub fn set_output_shape(&mut self, shape: &[usize]) {
        self.output.set_tensor_shape(Vec::<usize>::from(shape).as_ref());
    }
}