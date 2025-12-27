use ndarray::{ArrayD};
use crate::{
    memory::HostDeviceMem,
    atomics::{
        Taint,
        Guard
    },
    cu_rs::{
        stream::CuStream,
        error::{CuResult, CuError}
    }
};
use std::{
    cell::{RefCell, Ref, RefMut},
    collections::HashMap,
    marker::PhantomData,
    future::Future,
    sync::{
        Arc,
        atomic::{
            AtomicUsize,
            Ordering::SeqCst
        }
    }
};

pub struct UnsafeIOBuffers<'stream> {
    inputs: RefCell<HashMap<&'stream str,HostDeviceMem<'stream>>>,
    outputs: RefCell<HashMap<&'stream str,HostDeviceMem<'stream>>>,
    guard: Guard,
    taints: Arc<AtomicUsize>
}

unsafe impl<'stream> Send for UnsafeIOBuffers<'stream> {}
unsafe impl<'stream> Sync for UnsafeIOBuffers<'stream> {}

impl<'stream> UnsafeIOBuffers<'stream> {
    pub fn new(inputs: HashMap<&'stream str, usize>, outputs: HashMap<&'stream str, usize>, stream: CuStream) -> CuResult<Self> {
        let mut input_buffers = HashMap::new();
        let mut output_buffers = HashMap::new();
        for (name, size) in inputs {
            input_buffers.insert(name, HostDeviceMem::new(size, stream.clone())?);
        }
        for (name, size) in outputs {
            output_buffers.insert(name, HostDeviceMem::new(size, stream.clone())?);
        }
        Ok(Self {
            inputs: RefCell::new(input_buffers),
            outputs: RefCell::new(output_buffers),
            guard: Guard::new(),
            taints: Arc::new(AtomicUsize::new(0))
        })
    }

    pub fn input<'lock>(&'lock self, name: &str) -> CuResult<Ref<'lock, HostDeviceMem<'stream>>> {
        Ok(
            Ref::filter_map(self.inputs.borrow(), |inputs|
                inputs.get(name)
            ).map_err(|_| CuError::InvalidValue)?
        )
    }

    pub fn output<'lock>(&'lock self, name: &str) -> CuResult<Ref<'lock, HostDeviceMem<'stream>>> {
        Ok(
            Ref::filter_map(self.outputs.borrow(), |outputs|
                outputs.get(name)
            ).map_err(|_| CuError::InvalidValue)?
        )
    }

    pub fn input_mut<'lock>(&'lock self, name: &str) -> CuResult<RefMut<'lock, HostDeviceMem<'stream>>> {
        Ok(
            RefMut::filter_map(self.inputs.borrow_mut(), |inputs|
                inputs.get_mut(name)
            ).map_err(|_| CuError::InvalidValue)?
        )
    }

    pub fn output_mut<'lock>(&'lock self, name: &str) -> CuResult<RefMut<'lock, HostDeviceMem<'stream>>> {
        Ok(
            RefMut::filter_map(self.outputs.borrow_mut(), |outputs|
                outputs.get_mut(name)
            ).map_err(|_| CuError::InvalidValue)?
        )
    }

    pub fn taints(&self) -> usize {
        self.taints.load(SeqCst)
    }

    pub async fn push<'lock,T>(&'lock self, name: &str, src: &ArrayD<T>) -> CuResult<()>
    where T: Clone {
        let mut input = self.input_mut(name)?;
        input.load_ndarray(src)?;
        input.set_shape(&src.shape().to_vec().clone());
        match input.htod().await {
            Ok(_) => {
                Ok(())
            },
            Err(e) => {
                println!("PUSH Error: {:?}", e);
                Err(e)
            }
        }
    }

    // input device buffer to output device buffer (input to output device -> itod)
    pub async fn itod<'lock>(&'lock self, input_name: &str, output_name: &str) -> CuResult<()> {
        let input = self.input(input_name)?;
        let mut output = self.output_mut(output_name)?;
        output.set_shape(&input.shape().clone());
        match input.move_on_device(&mut output).await {
            Ok(_) => Ok(()),
            Err(e) => {
                println!("ITOD Error: {:?}", e);
                Err(e)
            }
        }
    }

    pub async fn pull<'lock,T>(&'lock self, output_name: &str) -> CuResult<ArrayD<T>>
    where T: Clone + Default
    {
        let mut output = self.output_mut(output_name)?;
        match output.dtoh().await {
            Ok(_) => Ok(output.dump_ndarray()),
            Err(e) => {
                println!("PULL Error: {:?}", e);
                Err(e)
            }
        }  
    }
}

pub struct IOBuffers<'stream> {
    buffers: UnsafeIOBuffers<'stream>,
    _phantom: PhantomData<&'stream ()>
}

impl<'stream> IOBuffers<'stream> {
    pub fn new(inputs: HashMap<&'stream str, usize>, outputs: HashMap<&'stream str, usize>, stream: CuStream) -> CuResult<Self> {
        Ok(Self {
            buffers: UnsafeIOBuffers::new(inputs, outputs, stream)?,
            _phantom: PhantomData
        })
    }

    pub fn taints(&self) -> usize {
        self.buffers.taints()
    }

    pub async fn with_guard<'buf, F, Fut, R>(&'buf self, size: usize, func: F) -> R
    where
        F: FnOnce(
            &'buf UnsafeIOBuffers<'stream>,
        ) -> Fut,
        Fut: Future<Output = R> + 'buf,
    {
        let _taint = Taint::new(self.buffers.taints.clone(), size);
        self.buffers.guard.with_lock( || func(&self.buffers)).await
    }
}

pub struct IOBufferPool<'stream> {
    pool: Vec<IOBuffers<'stream>>
}

impl<'stream> IOBufferPool<'stream> {
    pub fn new(num_buffers: usize, inputs: HashMap<&'stream str, usize>, outputs: HashMap<&'stream str, usize>, stream: CuStream) -> CuResult<Self> {
        let mut pool = Vec::new();
        for _ in 0..num_buffers {
            pool.push(IOBuffers::new(inputs.clone(), outputs.clone(), stream.clone())?);
        }
        Ok(Self {
            pool
        })
    }

    pub fn empty() -> Self {
        Self { pool: Vec::new() }
    }

    pub async fn with_buffers<'buf, F, Fut, R>(&'buf self, size: usize, func: F) -> R
    where
        F: FnOnce(&'buf UnsafeIOBuffers<'stream>) -> Fut,
        Fut: Future<Output = R> + 'buf,
    {
        // get the buffer with the least bytes in queue (taints) and pass the guard
        let buffer = self.pool.iter()
            .min_by_key(|buffer| buffer.taints())
            .expect("Empty buffer pool not supported!");
        buffer.with_guard(size, |buf| func(buf)).await
    }
}