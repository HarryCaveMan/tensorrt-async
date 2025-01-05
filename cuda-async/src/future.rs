use crate::{
    cu_api::cuLaunchHostFunc,
    cu_rs::{
        stream::CuStream,
        event::CuEvent,
        error::{CuResult, CuError}
    }
};
use std::{
    sync::Arc,
    future::Future,
    ffi::c_void,
    task::{Context, Poll, Waker},
    pin::Pin
};

pub struct CuEventFuture {
    event: Arc<CuEvent>,
    stream: Arc<CuStream>,
    waker: Arc<Option<Waker>>
}

impl CuEventFuture {
    pub fn new(event: CuEvent, stream: CuStream) -> Self {
        Self { 
            event: Arc::new(event),
            stream: Arc::new(stream),
            waker: Arc::new(None) 
        }
    }
}

impl Future for CuEventFuture {
    type Output = CuResult<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.event.query() {
            Ok(ready) => {
                if ready {
                    Poll::Ready(Ok(()))
                } else {
                    // Register callback to wake the task when event completes
                    let waker = Arc::new(Some(cx.waker().clone()));
                    unsafe {
                        cuLaunchHostFunc(
                            self.stream.get_raw(),
                            Some(wake_callback),
                            Arc::into_raw(waker) as *mut c_void
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
        let waker = Arc::from_raw(userData as *const Option<Waker>);
        if let Some(w) = &*waker {
            w.wake_by_ref();
        }
    }
}