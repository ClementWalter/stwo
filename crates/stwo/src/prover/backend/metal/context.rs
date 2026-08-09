//! Shared GPU device and command queue for all Metal kernel modules.
//!
//! A single queue keeps cross-stage submissions in order (a later stage can rely on an
//! earlier unwaited submission completing first), and avoids redundant device handles.
//! Metal devices and command queues are thread-safe; per-module mutable caches keep
//! their own locks.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use metal::{CommandBufferRef, CommandQueue, Device, MTLCommandBufferStatus};

static SUCCESSFUL_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static FAILED_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SubmissionCounters {
    pub successful: u64,
    pub failed: u64,
}

impl SubmissionCounters {
    /// Computes a session-local delta across a possible `u64` counter wrap.
    pub(crate) const fn delta_from(self, start: Self) -> Self {
        Self {
            successful: self.successful.wrapping_sub(start.successful),
            failed: self.failed.wrapping_sub(start.failed),
        }
    }
}

/// Returns the process-lifetime checked-command counters. They are never reset;
/// sessions take snapshots and report wrap-safe deltas.
pub(crate) fn submission_counters() -> SubmissionCounters {
    SubmissionCounters {
        successful: SUCCESSFUL_SUBMISSIONS.load(Ordering::Relaxed),
        failed: FAILED_SUBMISSIONS.load(Ordering::Relaxed),
    }
}

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
        SUCCESSFUL_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    } else {
        FAILED_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
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

    #[test]
    fn submission_delta_is_wrap_safe() {
        let start = super::SubmissionCounters {
            successful: u64::MAX - 1,
            failed: u64::MAX,
        };
        let end = super::SubmissionCounters {
            successful: 1,
            failed: 0,
        };
        assert_eq!(
            end.delta_from(start),
            super::SubmissionCounters {
                successful: 3,
                failed: 1,
            }
        );
    }
}
