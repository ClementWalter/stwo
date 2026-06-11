//! Shared GPU device and command queue for all Metal kernel modules.
//!
//! A single queue keeps cross-stage submissions in order (a later stage can rely on an
//! earlier unwaited submission completing first), and avoids redundant device handles.
//! Metal devices and command queues are thread-safe; per-module mutable caches keep
//! their own locks.

use std::sync::OnceLock;

use metal::{CommandQueue, Device};

pub(crate) struct Gpu {
    pub device: Device,
    pub queue: CommandQueue,
}

// Metal devices and command queues are documented thread-safe.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// The process-wide GPU handle; `None` when no usable unified-memory device exists.
pub(crate) fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| {
        let device = Device::system_default()?;
        if !device.has_unified_memory() {
            return None;
        }
        let queue = device.new_command_queue();
        Some(Gpu { device, queue })
    })
    .as_ref()
}
