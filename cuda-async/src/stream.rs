use crate::{
    io_buffers::{
        IOBufferPool,
        UnsafeIOBuffers
    },
    atomics::{
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
    pin::Pin,
    collections::HashMap,
    future::Future,
    sync::{
        Arc,
        atomic::{
            AtomicUsize,
            Ordering::SeqCst
        }
    }
};


pub struct CuStreamWithBuffers<'context> {
    stream: CuStream,
    buffers: IOBufferPool<'context>,
    taints: Arc<AtomicUsize>
}

impl<'context> CuStreamWithBuffers<'context> {
    pub fn new(num_buffers: usize, inputs: HashMap<&'context str, usize>, outputs: HashMap<&'context str, usize>) -> CuResult<Self> {
        let stream = CuStream::new()?;
        let buffers = IOBufferPool::new(
            num_buffers,
            inputs,
            outputs,
            stream.clone()
        )?;
        Ok(Self {
            stream: stream.clone(),
            buffers: buffers,
            taints: Arc::new(AtomicUsize::new(0))
        })
    }

    pub async fn with_buffers<'buf, F, Fut, R>(&'buf self, size: usize, func: F) -> R
    where
        F: FnOnce(&'buf UnsafeIOBuffers<'context>) -> Fut,
        Fut: Future<Output = R> + 'buf
    {
        let _taint = Taint::new(self.taints.clone(), size);
        self.buffers.with_buffers(size, |buf| func(buf)).await
    }

    pub fn taints(&self) -> usize {
        self.taints.load(SeqCst)
    }
}

pub struct CuStreamPool<'context> {
    streams: Vec<CuStreamWithBuffers<'context>>
}

impl<'context> CuStreamPool<'context> {
    pub fn new(num_streams: usize, buffers_per_stream: usize, inputs: HashMap<&'context str, usize>, outputs: HashMap<&'context str, usize>) -> CuResult<Self> {
        let mut streams = Vec::new();
        for _ in 0..num_streams {
            streams.push(CuStreamWithBuffers::new(buffers_per_stream, inputs.clone(), outputs.clone())?);
        }
        Ok(Self {
            streams
        })
    }

    pub async fn stream_with_buffers<'buf, F, Fut, R>(&'buf self, size: usize, func: F) -> R
    where
        F: FnOnce(&'buf UnsafeIOBuffers<'context>) -> Fut,
        Fut: Future<Output = R> + 'buf 
    {
        let stream = self.streams.iter().min_by_key(|stream| stream.taints()).expect("Empty stream pool not supported!");
        stream.with_buffers(size, |buf| func(buf)).await
    }
}