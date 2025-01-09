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

pub struct CuEventFuture<'context> {
    event: CuEvent,
    stream: CuStream,
    // exists purely to hold the lifetime prameter
    _phantom: PhantomData<&'context ()>
}

impl<'context> CuEventFuture<'context> {
    pub fn new(event: CuEvent, stream: CuStream) -> Self {
        Self { 
            event: event,
            stream: stream,
            _phantom: PhantomData
        }
    }
}

impl<'context> Future for CuEventFuture<'context> {
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