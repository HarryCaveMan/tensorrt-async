use ndarray::{ArrayD,IxDyn};
use crate::{
    cu_api::{
        CUdeviceptr,
        CUresult,
        cuMemAlloc_v2,
        cuMemAllocHost_v2,
        cuMemcpyDtoHAsync_v2,
        cuMemcpyHtoDAsync_v2,
        cuMemcpyDtoDAsync_v2,
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
    size: usize,
    tensor_shape: Vec<usize>,
    stream: CuStream,
    htod_event: CuEvent,
    dtoh_event: CuEvent,
    dtod_event: CuEvent
}

impl HostDeviceMem {
    pub fn new<T>(size:usize,stream: &CuStream) -> CuResult<Self> 
    {
        let mut host_ptr: *mut c_void = null_mut();
        let mut device_ptr: CUdeviceptr = 0;
        let host_alloc_res: CUresult = unsafe {
            cuMemAllocHost_v2(&mut host_ptr, size)
        };
        let htod_event: CuEvent = CuEvent::new()?;
        let dtoh_event: CuEvent = CuEvent::new()?;
        let dtod_event: CuEvent = CuEvent::new()?;
        match wrap!((), host_alloc_res) {
            Ok(_) => {
                let device_alloc_res: CUresult = unsafe {
                    cuMemAlloc_v2(&mut device_ptr, size)
                };
                wrap!(
                    Self {
                        host_ptr: host_ptr, 
                        device_ptr: device_ptr,
                        size: size,
                        tensor_shape: Vec::<usize>::new(),
                        stream: stream.clone(),
                        htod_event: htod_event,
                        dtoh_event: dtoh_event,
                        dtod_event: dtod_event
                    },
                    device_alloc_res
                )
            }
            Err(cuErr) => Err(cuErr)
        }
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

    pub fn tensor_shape(&self) -> &Vec<usize> {
        &self.tensor_shape
    }

    pub fn set_tensor_shape(&mut self, shape: &Vec<usize>) {
        self.tensor_shape = shape.clone();
    }

    pub fn load_ndarray<T>(&mut self, src: &ArrayD<T>) -> CuResult<()>
    where T: Clone
    {
        let size = src.len() * size_of::<T>();
        if size > self.size {
            return Err(CuError::InvalidValue);
        }
        self.set_tensor_shape(&Vec::<usize>::from(src.shape()));
        unsafe {
            copy_nonoverlapping(
                src.as_ptr() as *const c_void,
                self.host_ptr,
                size
            );
        }
        Ok(())
    }

    pub fn dump_ndarray<T>(&self) -> ArrayD<T>
    where T: Clone + Default 
    {        
        let mut array: ArrayD<T> = ArrayD::<T>::default(IxDyn(&self.tensor_shape));
        let array_size: usize = array.len() * size_of::<T>();
        unsafe {
            copy_nonoverlapping(
                self.host_ptr,
                array.as_mut_ptr() as *mut c_void,
                array_size
            );
        }
        array
    }

    pub unsafe fn htod_async(&self) -> CUresult 
    {
        let htod_res: CUresult = cuMemcpyHtoDAsync_v2(
                self.device_ptr,
                self.host_ptr,
                self.size,
                self.stream.get_raw()
        );
        self.htod_event.record(&self.stream);
        htod_res
    }

    pub async fn htod(&self) -> CuResult<()>
    {
        let htod_async_res: CUresult = unsafe { self.htod_async() };
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

    pub unsafe fn dtoh_async(&self) -> CUresult
    {
        let dtoh_res: CUresult = cuMemcpyDtoHAsync_v2(
                self.host_ptr,
                self.device_ptr,
                self.size,
                self.stream.get_raw()
        );
        self.dtoh_event.record(&self.stream);
        dtoh_res
    }

    pub async fn dtoh(&self) -> CuResult<()>
    {
        let  dtoh_async_res: CUresult = unsafe { self.dtoh_async() };
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

    pub unsafe fn move_on_device_async(&self, dst: &mut HostDeviceMem) -> CUresult
    {
        let dtod_result: CUresult = cuMemcpyDtoDAsync_v2(
                dst.device_ptr(),
                self.device_ptr,
                self.size,
                self.stream.get_raw()
        );
        self.dtod_event.record(&self.stream);
        dtod_result
    }

    pub async fn move_on_device(&self, dst: &mut HostDeviceMem) -> CuResult<()>
    {
        let dtod_async_res: CUresult = unsafe { self.move_on_device_async(dst) };
        let event: CuEvent = self.dtod_event.clone();
        let stream: CuStream = self.stream.clone();
        match wrap_async!(
            dtod_async_res,
            event,
            stream
        )
        {
            Ok(future) => {
                future.await;
                dst.set_tensor_shape(&self.tensor_shape);
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