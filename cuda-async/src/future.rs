use crate::{
    cu_abi::cuLaunchHostFunc,
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

/*
This is the lowest cost Futere implementation I could think of.
First, it checks if the event has already occured, if so, it returns Poll::Ready(Ok(()))
If the event has not occured, it:
  - Registers wake_callback with cuLacucnHostFunc as a c ABI function pointer
  - Passes the waker through to the callback through the userData pointer
  - Releases the context to the event loop
  - consumes no CPU time and only a few pointers/refs while waiting
  - Wakes up only when the interrupt-triggered callback is invoked
*/

pub struct CuEventFuture<'context> {
    event: &'context CuEvent,
    stream: &'context CuStream,
    // exists purely to hold the lifetime prameter
    _phantom: PhantomData<&'context ()>
}

impl<'context> CuEventFuture<'context> {
    pub fn new(event: &'context CuEvent, stream: &'context CuStream) -> Self {
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
                    unsafe {
                        cuLaunchHostFunc(
                            self.stream.get_raw(),
                            Some(wake_callback),
                            cx.waker() as *const _ as *mut c_void
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