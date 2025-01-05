use ndarray::{ArrayD,IxDyn};
use tokio::sync::{Mutex,OwnedMutexGuard};
use crate::{
    taint::Taint,
    cu_api::{
        CUdeviceptr,
        CUresult,
        cuMemAlloc_v2,
        cuMemAllocHost_v2,
        cuMemcpyDtoHAsync_v2,
        cuMemcpyHtoDAsync_v2,
        cuMemFree_v2,
        cuMemFreeHost,
        cuEventDestroy_v2
    },
    cu_rs::{
        stream::CuStream,
        event::CuEvent,
        error::{CuResult, CuError}
    }
};

use std::{
    sync::{
        Arc,
        atomic::{
            AtomicUsize,
            Ordering::SeqCst
        }
    },
    future::Future,
    ffi::c_void,
    ptr::{
        null_mut,
        copy_nonoverlapping
    },
    mem::size_of,
    slice::{
        from_raw_parts,
        from_raw_parts_mut
    }
};

pub struct HostDeviceMem {    
    host_ptr: *mut c_void,
    device_ptr: CUdeviceptr,
    max_shape: Vec<usize>,
    size: usize,
    stream: CuStream,
    htod_event: CuEvent,
    dtoh_event: CuEvent,
    guard: Arc<Mutex<()>>,
    taints: AtomicUsize
}

impl HostDeviceMem {
    pub fn new<T>(max_shape: &[usize],stream: &CuStream) -> CuResult<Self> 
    {
        let size: usize = max_shape.iter().product::<usize>() * size_of::<T>();
        let mut host_ptr: *mut c_void = null_mut();
        let mut device_ptr: CUdeviceptr = 0;
        let host_alloc_res: CUresult = unsafe {
            cuMemAllocHost_v2(&mut host_ptr, size)
        };
        let htod_event: CuEvent = CuEvent::new()?;
        let dtoh_event: CuEvent = CuEvent::new()?;
        match wrap!((), host_alloc_res) {
            Ok(_) => {
                let device_alloc_res: CUresult = unsafe {
                    cuMemAlloc_v2(&mut device_ptr, size)
                };
                wrap!(
                    Self {
                        host_ptr: host_ptr, 
                        device_ptr: device_ptr,
                        max_shape: Vec::from(max_shape),
                        size: size, 
                        stream: stream.clone(),
                        htod_event: htod_event,
                        dtoh_event: dtoh_event,
                        guard: Arc::new(Mutex::new(())),
                        taints: AtomicUsize::new(0)
                    },
                    device_alloc_res
                )
            }
            Err(cuErr) => Err(cuErr)
        }
    }

    pub fn taints(&self) -> usize {
        self.taints.load(SeqCst)
    }

    pub async fn with_guard<F, Fut, R>(&self, func: F) -> R
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = R>,
    {
        let _taint = Taint::new(&self.taints);
        let _guard = self.guard.clone().lock_owned().await;
        func().await
    }

    pub unsafe fn host_ptr(&self) -> *mut c_void
    {
        self.host_ptr
    }

    pub unsafe fn device_ptr(&self) -> CUdeviceptr
    {
        self.device_ptr
    }

    pub fn host_as_slice<T>(&self) -> &[T] {
        let size = self.size / size_of::<T>();
        unsafe { from_raw_parts(self.host_ptr as *const T, size) }
    }

    pub fn host_as_mut_slice<T>(&mut self) -> &mut [T] {
        let size = self.size / size_of::<T>();
        unsafe { from_raw_parts_mut(self.host_ptr as *mut T, size) }
    }

    pub fn load_ndarray<T>(&self, src: &ArrayD<T>) -> CuResult<()>
    where T: Clone
    {
        let size = src.len() * size_of::<T>();
        if size > self.size {
            return Err(CuError::InvalidValue);
        }        
        unsafe {
            copy_nonoverlapping(
                src.as_ptr() as *const c_void,
                self.host_ptr,
                size
            );
        }
        Ok(())
    }

    pub fn dump_ndarray<T>(&self,shape: &[usize]) -> ArrayD<T>
    where T: Clone + Default 
    {
        let size = self.size / size_of::<T>();
        let mut array = ArrayD::<T>::default(IxDyn(shape));
        unsafe {
            copy_nonoverlapping(
                self.host_ptr,
                array.as_mut_ptr() as *mut c_void,
                self.size
            );
        }
        array
    }

    pub fn htod_async(&self) -> CUresult 
    {
        let htod_res: CUresult = unsafe {
            cuMemcpyHtoDAsync_v2(
                self.device_ptr,
                self.host_ptr,
                self.size,
                self.stream.get_raw()
            )
        };
        self.htod_event.record(&self.stream);
        htod_res
    }

    pub async fn htod(&self) -> CuResult<()>
    {
        let htod_async_res: CUresult = self.htod_async();
        let event: CuEvent = self.htod_event.clone();
        let stream: CuStream = self.stream.clone();
        match wrap_async!(
            htod_async_res,
            event,
            stream
        )
        {
            Ok(future) => {
                future.await;
                Ok(())
            }
            Err(cuErr) => Err(cuErr)
        }
    }

    pub fn dtoh_async(&self) -> CUresult
    {
        let dtoh_res: CUresult = unsafe {
            cuMemcpyDtoHAsync_v2(
                self.host_ptr,
                self.device_ptr,
                self.size,
                self.stream.get_raw()
            )
        };
        self.dtoh_event.record(&self.stream);
        dtoh_res
    }

    pub async fn dtoh(&self) -> CuResult<()>
    {
        let  dtoh_async_res: CUresult = self.dtoh_async();
        let event: CuEvent = self.dtoh_event.clone();
        let stream: CuStream = self.stream.clone();
        match wrap_async!(
            dtoh_async_res,
            event,
            stream
        )
        {
            Ok(future) => {
                future.await;
                Ok(())
            }
            Err(cuErr) => Err(cuErr)
        }
    }
}

impl Drop for HostDeviceMem {
    fn drop(&mut self) 
    {
        unsafe { 
            cuMemFreeHost(self.host_ptr);
            cuMemFree_v2(self.device_ptr);
        };
    }
}