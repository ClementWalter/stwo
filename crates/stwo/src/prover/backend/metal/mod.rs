//! Apple-GPU (Metal) kernels accelerating prover hot paths on unified-memory
//! machines.
//!
//! Metal is not a separate proving backend. [`super::CpuBackend`] remains the
//! backend and dispatches eligible, sufficiently large work to these kernels. Small
//! or unsupported shapes are expected to stay on the CPU, whose implementations
//! also remain the reference-test ground truths.
//!
//! [`MetalSession`] provides opt-in, process-local evidence that a proof actually
//! participated in checked Metal work. Its mutex serializes callers that all use this
//! API. It cannot isolate callers that bypass it: every checked wait contributes to
//! the process-global counters, so concurrent CPU-backend/Metal work outside the
//! session contaminates its delta and invalidates attribution. Such work must not run
//! concurrently when the report is used as evidence.
//! Admission validates the unified-memory device and every static pipeline before a
//! caller starts its transcript. Constraint pipelines are deliberately excluded:
//! their Metal source is generated dynamically from the proof's AIR and therefore
//! cannot be compiled during static preflight.

use std::fmt;
use std::sync::{Mutex, MutexGuard};

pub(crate) mod blake2s;
pub(crate) mod commit;
pub mod constraints;
pub(crate) mod context;
pub(crate) mod fft;
pub(crate) mod fri;
pub(crate) mod ood;
pub(crate) mod quotients;
pub(crate) mod twiddles;

static PROOF_SESSION_MUTEX: Mutex<()> = Mutex::new(());

/// A process-local report of checked Metal command buffers submitted during a
/// [`MetalSession`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalSessionReport {
    pub device_name: String,
    pub successful_submissions: u64,
    pub failed_submissions: u64,
}

/// Failure to admit a proof into a Metal session before transcript construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalAdmissionError {
    /// Another session panicked while holding the process-wide proof mutex.
    SessionMutexPoisoned,
    /// The process has no Metal device backed by unified memory.
    NoUnifiedMemoryDevice,
    /// A static kernel module could not initialize all of its pipelines.
    StaticPipelineUnavailable { module: &'static str },
}

impl fmt::Display for MetalAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionMutexPoisoned => {
                write!(formatter, "the process-wide Metal proof mutex is poisoned")
            }
            Self::NoUnifiedMemoryDevice => {
                write!(formatter, "no unified-memory Metal device is available")
            }
            Self::StaticPipelineUnavailable { module } => {
                write!(
                    formatter,
                    "the static Metal {module} pipeline is unavailable"
                )
            }
        }
    }
}

impl std::error::Error for MetalAdmissionError {}

/// An admitted interval used to attribute checked Metal submissions to one proof.
///
/// Call [`Self::admit`] before creating the proof transcript and [`Self::finish`]
/// after proving. The process-wide mutex is held for that entire interval.
pub struct MetalSession {
    guard: MutexGuard<'static, ()>,
    device_name: String,
    starting_counters: context::SubmissionCounters,
}

impl MetalSession {
    /// Locks the process-wide proof session, validates all static Metal resources,
    /// then snapshots the checked-command counters.
    ///
    /// Pipeline compilation performed by admission is outside the reported interval.
    /// Dynamic AIR constraint pipelines are compiled later and are not part of this
    /// preflight.
    pub fn admit() -> Result<Self, MetalAdmissionError> {
        let guard = lock_session_mutex(&PROOF_SESSION_MUTEX)?;
        let gpu = context::gpu().ok_or(MetalAdmissionError::NoUnifiedMemoryDevice)?;
        validate_static_pipelines()?;

        Ok(Self {
            guard,
            device_name: gpu.device.name().to_owned(),
            starting_counters: context::submission_counters(),
        })
    }

    /// Ends the attributed interval and reports its checked-command counter deltas.
    pub fn finish(self) -> MetalSessionReport {
        let delta = context::submission_counters().delta_from(self.starting_counters);
        let report = MetalSessionReport {
            device_name: self.device_name,
            successful_submissions: delta.successful,
            failed_submissions: delta.failed,
        };
        drop(self.guard);
        report
    }
}

fn lock_session_mutex(mutex: &Mutex<()>) -> Result<MutexGuard<'_, ()>, MetalAdmissionError> {
    mutex
        .lock()
        .map_err(|_| MetalAdmissionError::SessionMutexPoisoned)
}

fn validate_static_pipelines() -> Result<(), MetalAdmissionError> {
    type StaticPipelineProbe = (&'static str, fn() -> bool);
    let modules: [StaticPipelineProbe; 6] = [
        ("Blake2s", blake2s::is_ready),
        ("FFT", fft::is_ready),
        ("quotient", quotients::is_ready),
        ("twiddle", twiddles::is_ready),
        ("FRI", fri::is_ready),
        ("out-of-domain", ood::is_ready),
    ];
    for (module, is_ready) in modules {
        if !is_ready() {
            return Err(MetalAdmissionError::StaticPipelineUnavailable { module });
        }
    }
    Ok(())
}

/// Policy applied to a completed [`MetalSessionReport`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalRequirement {
    /// Accept CPU-only placement and any observed Metal outcome.
    Prefer,
    /// Require at least one checked successful submission and no checked failures.
    ///
    /// This proves Metal participation, not that every operation ran on Metal.
    /// Hybrid proofs are expected to retain small or unsupported work on the CPU.
    ParticipationRequired,
}

impl MetalRequirement {
    pub fn validate(self, report: &MetalSessionReport) -> Result<(), MetalRequirementError> {
        if self == Self::Prefer {
            return Ok(());
        }
        if report.failed_submissions != 0 {
            return Err(MetalRequirementError::FailedSubmissions {
                count: report.failed_submissions,
            });
        }
        if report.successful_submissions == 0 {
            return Err(MetalRequirementError::NoSuccessfulSubmissions);
        }
        Ok(())
    }
}

/// Failure to satisfy a Metal participation policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetalRequirementError {
    /// No checked Metal command completed successfully during the session.
    NoSuccessfulSubmissions,
    /// One or more checked Metal commands failed during the session.
    FailedSubmissions { count: u64 },
}

impl fmt::Display for MetalRequirementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuccessfulSubmissions => {
                write!(formatter, "the proof made no successful Metal submissions")
            }
            Self::FailedSubmissions { count } => {
                write!(formatter, "the proof had {count} failed Metal submissions")
            }
        }
    }
}

impl std::error::Error for MetalRequirementError {}

/// Initializes the GPU device and compiles the kernel pipelines (a one-time cost of
/// ~100ms) so the first commitment doesn't pay it. Safe to call from any thread; a
/// no-op when no usable device exists.
pub fn warmup() {
    blake2s::warmup();
    fft::warmup();
    quotients::warmup();
    twiddles::warmup();
    fri::warmup();
    ood::warmup();
}

/// Waits for an externally encoded Metal command buffer and records its result in
/// the same checked-submission counters used by [`MetalSession`].
///
/// Callers must invoke this exactly once after committing the command buffer and
/// must not read outputs or fall back to mutation-sensitive host work on error.
pub fn record_checked_completion(
    command_buffer: &metal::CommandBufferRef,
) -> Result<(), metal::MTLCommandBufferStatus> {
    context::wait_for_completion(command_buffer)
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Duration;

    use super::{MetalRequirement, MetalRequirementError, MetalSessionReport};

    fn report(successful_submissions: u64, failed_submissions: u64) -> MetalSessionReport {
        MetalSessionReport {
            device_name: "test device".to_owned(),
            successful_submissions,
            failed_submissions,
        }
    }

    #[test]
    fn prefer_accepts_cpu_only_and_failed_sessions() {
        assert_eq!(MetalRequirement::Prefer.validate(&report(0, 0)), Ok(()));
        assert_eq!(MetalRequirement::Prefer.validate(&report(2, 1)), Ok(()));
    }

    #[test]
    fn participation_requires_a_successful_submission() {
        assert_eq!(
            MetalRequirement::ParticipationRequired.validate(&report(0, 0)),
            Err(MetalRequirementError::NoSuccessfulSubmissions)
        );
        assert_eq!(
            MetalRequirement::ParticipationRequired.validate(&report(1, 0)),
            Ok(())
        );
    }

    #[test]
    fn participation_rejects_any_failed_submission() {
        assert_eq!(
            MetalRequirement::ParticipationRequired.validate(&report(3, 2)),
            Err(MetalRequirementError::FailedSubmissions { count: 2 })
        );
    }

    #[test]
    fn session_mutex_serializes_participants() {
        let mutex = Arc::new(Mutex::new(()));
        let guard = super::lock_session_mutex(&mutex).unwrap();
        let worker_mutex = Arc::clone(&mutex);
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let _guard = super::lock_session_mutex(&worker_mutex).unwrap();
            acquired_tx.send(()).unwrap();
        });

        started_rx.recv().unwrap();
        assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());
        drop(guard);
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn session_mutex_rejects_poison() {
        let mutex = Arc::new(Mutex::new(()));
        let worker_mutex = Arc::clone(&mutex);
        assert!(std::thread::spawn(move || {
            let _guard = worker_mutex.lock().unwrap();
            panic!("poison test mutex");
        })
        .join()
        .is_err());

        assert!(matches!(
            super::lock_session_mutex(&mutex),
            Err(super::MetalAdmissionError::SessionMutexPoisoned)
        ));
    }
}
