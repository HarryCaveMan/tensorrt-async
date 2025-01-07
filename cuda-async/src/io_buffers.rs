use tokio::sync::{Mutex,OwnedMutexGuard};
use ndarray::{ArrayD};
use crate::{
    memory::HostDeviceMem,
    taint::Taint,
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

pub struct IOBuffers {
    input: HostDeviceMem,
    output: HostDeviceMem,
    guard: Arc<Mutex<()>>,
    taints: Arc<AtomicUsize>
}

impl IOBuffers {
    pub fn new<T>(input_size: usize, output_size: usize, stream: &CuStream) -> CuResult<Self> {
        let input = HostDeviceMem::new::<T>(input_size, stream)?;
        let output = HostDeviceMem::new::<T>(output_size, stream)?;
        let guard = Arc::new(Mutex::new(()));
        Ok(Self {
            input,
            output,
            guard,
            taints: AtomicUsize::new(0).into()
        })
    }

    pub async fn with_guard<F, Fut, R>(&self, size: usize, func: F) -> R
    where
        F: FnOnce(&Self) -> Fut,
        Fut: Future<Output = R>,
    {
        let _taint = Taint::new(Arc::clone(&self.taints), size);
        let _guard = self.guard.clone().lock_owned().await;
        func(&self).await
    }

    pub async fn with_guard_mut<'a, F, Fut, R>(&'a mut self, size: usize, func: F) -> R
    where
        F: FnOnce(&'a mut Self) -> Fut,
        Fut: Future<Output = R> + 'a,
    {
        let _taint = Taint::new(Arc::clone(&self.taints), size);
        let _guard = self.guard.clone().lock_owned().await;
        func(self).await
    }    

    pub fn input(&self) -> &HostDeviceMem {
        &self.input
    }

    pub fn output(&self) -> &HostDeviceMem {
        &self.output
    }

    pub fn input_mut(&mut self) -> &mut HostDeviceMem {
        &mut self.input
    }

    pub fn output_mut(&mut self) -> &mut HostDeviceMem {
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