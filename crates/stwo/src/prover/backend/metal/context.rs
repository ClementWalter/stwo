//! Shared GPU device and command queue for all Metal kernel modules.
//!
//! A single queue keeps cross-stage submissions in order (a later stage can rely on an
//! earlier unwaited submission completing first), and avoids redundant device handles.
//! Metal devices and command queues are thread-safe; per-module mutable caches keep
//! their own locks.

use std::sync::OnceLock;

use metal::{CommandBufferRef, CommandQueue, Device, MTLCommandBufferStatus};

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

/// Waits for a submitted command buffer and rejects every terminal state except
/// `Completed`.
///
/// `metal-rs` 0.33 exposes the command status but not Metal's accompanying
/// `NSError`, so the status is the most precise portable diagnostic available here.
/// Callers must check this result before reading GPU outputs. If a command may have
/// mutated data needed for a CPU fallback, callers must not attempt that fallback
/// after an error unless they retained an untouched copy.
pub(crate) fn wait_for_completion(
    command_buffer: &CommandBufferRef,
) -> Result<(), MTLCommandBufferStatus> {
    command_buffer.wait_until_completed();
    let status = command_buffer.status();
    if completion_succeeded(status) {
        Ok(())
    } else {
        tracing::error!(
            ?status,
            "Metal command buffer did not complete successfully"
        );
        Err(status)
    }
}

fn completion_succeeded(status: MTLCommandBufferStatus) -> bool {
    status == MTLCommandBufferStatus::Completed
}

#[cfg(test)]
mod tests {
    use metal::MTLCommandBufferStatus;

    #[test]
    fn only_completed_status_is_successful() {
        assert!(super::completion_succeeded(
            MTLCommandBufferStatus::Completed
        ));
        for status in [
            MTLCommandBufferStatus::NotEnqueued,
            MTLCommandBufferStatus::Enqueued,
            MTLCommandBufferStatus::Committed,
            MTLCommandBufferStatus::Scheduled,
            MTLCommandBufferStatus::Error,
        ] {
            assert!(!super::completion_succeeded(status));
        }
    }
}
