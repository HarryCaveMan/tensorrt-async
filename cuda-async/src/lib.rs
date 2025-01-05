#[macro_use]
extern crate enum_primitive;

extern crate cuda_rs_sys as cu_api;

#[macro_use]
extern crate cuda_rs as cu_rs;

#[macro_use]
mod macros;

pub (crate) mod future;


// public interface
pub mod memory;
pub mod taint;
pub use cu_rs::{
    init as cu_init,
    context::CuContext,
    device::CuDevice,
    error::{
        CuResult,
        CuError
    },
    stream::CuStream,
    event::CuEvent
};
pub use future::CuEventFuture;