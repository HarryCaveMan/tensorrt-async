use crate::{
    cu_ffi::cuLaunchHostFunc,
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
    pin::Pin,
    sync::{Arc, OnceLock},
    sync::atomic::{AtomicU8, Ordering}
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

const STATUS_NONE: u8 = 0;
const STATUS_REGISTERED: u8 = 1;
const STATUS_CANCELLED: u8 = 2;

struct UserData {
    waker: OnceLock<Waker>,
    status: AtomicU8
}

pub struct CuEventFuture<'context> {
    event: &'context CuEvent,
    stream: &'context CuStream,
    user_data: Arc<UserData>
}

impl<'context> CuEventFuture<'context> {
    pub fn new(event: &'context CuEvent, stream: &'context CuStream) -> Self {
        let mut user_data = Arc::new(UserData {
            waker: OnceLock::new(), // Placeholder, will be set in poll
            status: AtomicU8::new(STATUS_NONE)
        });
        Self { 
            event: event,
            stream: stream,
            user_data: user_data
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
                } else if self.user_data.status.load(Ordering::SeqCst) == STATUS_NONE {
                    // Register callback to wake the task when event completes
                    self.user_data.waker.set(cx.waker().clone()).ok();
                    unsafe {
                        cuLaunchHostFunc(
                            self.stream.get_raw(),
                            Some(wake_callback),
                            Arc::into_raw(self.user_data.clone()) as *mut c_void
                        );
                    }
                    self.user_data.status.store(STATUS_REGISTERED, Ordering::SeqCst);
                }
                // The following should only execute once per task
                // The event will always query true after the callback is invoked
                // And the pending task will not be polled until the callback invokes the waker
                Poll::Pending
                
            }
            Err(CuError) => Poll::Ready(Err(CuError))
        }
    }
}

impl Drop for CuEventFuture<'_> {
    fn drop(&mut self) {
        self.user_data.status.compare_exchange(
            STATUS_REGISTERED,
            STATUS_CANCELLED,
            Ordering::SeqCst,
            Ordering::SeqCst
        ).ok();
    }
}

extern "C" fn wake_callback(userData: *mut c_void) {
    unsafe {
        let user_data = Arc::from_raw(userData as *const UserData);
        if user_data.status.load(Ordering::SeqCst) == STATUS_REGISTERED {
            if let Some(waker) = user_data.waker.get() {
                waker.wake_by_ref();
            }
        }
    }
}