use std::sync::atomic::{
    AtomicUsize,
    Ordering::SeqCst
};

pub struct Taint<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> Taint<'a> {
    pub fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, SeqCst);
        Self { counter }
    }
}

impl<'a> Drop for Taint<'a> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, SeqCst);
    }
}