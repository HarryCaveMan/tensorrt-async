macro_rules! wrap {
    ($val:expr, $res:ident) => (
        if $res == crate::cu_abi::cudaError_enum_CUDA_SUCCESS {
            Ok($val)
        } else {
            use crate::cu_rs::error::CuError;
            Err(CuError::from($res))
        }
    )
}

macro_rules! wrap_async {
    ($res:expr,$cuEventClone:ident,$cuStreamClone:ident) => (
        if $res == crate::cu_abi::cudaError_enum_CUDA_SUCCESS {
            use crate::future::CuEventFuture;
            Ok(CuEventFuture::new($cuEventClone,$cuStreamClone))
        }
        else {
            use crate::cu_rs::error::CuError;
            Err(CuError::from($res))
        }
    )
}