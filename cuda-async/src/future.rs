use crate::{
    cu_api::cuLaunchHostFunc,
    cu_rs::{
        stream::CuStream,
        event::CuEvent,
        error::{CuResult, CuError}
    }
};
use std::{
    future::Future,
    ffi::c_void,
    task::{Context, Poll, Waker},
    marker::PhantomData,
    pin::Pin
};

pub struct CuEventFuture<'a> {
    event: CuEvent,
    stream: CuStream,
    _phantom: PhantomData<&'a ()>
}

impl<'a> CuEventFuture<'a> {
    pub fn new(event: CuEvent, stream: CuStream) -> Self {
        Self { 
            event: event,
            stream: stream,
            _phantom: PhantomData
        }
    }
}

impl<'a> Future for CuEventFuture<'a> {
    type Output = CuResult<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.event.query() {
            Ok(ready) => {
                if ready {
                    Poll::Ready(Ok(()))
                } else {
                    // Register callback to wake the task when event completes
                    let waker = cx.waker().clone();
                    unsafe {
                        cuLaunchHostFunc(
                            self.stream.get_raw(),
                            Some(wake_callback),
                            &waker as *const _ as *mut c_void
                        );
                    }
                    Poll::Pending
                }
            }
            Err(CuError) => Poll::Ready(Err(CuError))
        }
    }
}

extern "C" fn wake_callback(userData: *mut c_void) {
    unsafe {
        let waker = userData as *const Waker;
        if let w = &*waker {
            w.wake_by_ref();
        }
    }
}