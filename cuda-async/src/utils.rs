use tokio::sync::{Mutex,OwnedMutexGuard};
use std::sync::{
    Arc,
    atomic::{
        AtomicUsize,
        Ordering::SeqCst
    }
};

pub struct Taint {
    counter: Arc<AtomicUsize>,
    size: usize
}

impl Taint {
    pub fn new(counter: Arc<AtomicUsize>,size: usize) -> Self {
        counter.fetch_add(size, SeqCst);
        Self { counter, size }
    }
}

impl Drop for Taint {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.size, SeqCst);
    }
}

pub struct Guard {
    mutex: Arc<Mutex<()>>
}

impl Guard {
    pub fn new() -> Self {
        Self {
            mutex: Arc::new(Mutex::new(()))
        }
    }

    pub async fn lock(&self) -> OwnedMutexGuard<()> {
        self.mutex.clone().lock_owned().await
    }
}