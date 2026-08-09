use std::fmt;

/// A failure discovered only after the quotient command was submitted. Callers must
/// treat every variant as terminal: output buffers may already have been mutated and
/// are never eligible for CPU fallback.
#[derive(Debug)]
pub(crate) enum QuotientMetalError {
    NumeratorCommandFailed {
        status: metal::MTLCommandBufferStatus,
        batch_count: usize,
        dispatch_count: usize,
    },
    CommandFailed {
        status: metal::MTLCommandBufferStatus,
    },
    Pole {
        sample_mask: u32,
        first_row: usize,
        sample_first_rows: [Option<usize>; 6],
    },
}

impl fmt::Display for QuotientMetalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NumeratorCommandFailed {
                status,
                batch_count,
                dispatch_count,
            } => write!(
                formatter,
                "Metal quotient-numerator command failed with status {status:?} after submitting \
                 {batch_count} stable batches in {dispatch_count} ordered dispatches"
            ),
            Self::CommandFailed { status } => write!(
                formatter,
                "Metal quotient command failed with status {status:?}"
            ),
            Self::Pole {
                sample_mask,
                first_row,
                sample_first_rows,
            } => write!(
                formatter,
                "Metal quotient denominator pole: sample mask {sample_mask:#08b}, first row \
                 {first_row}, per-sample first rows {sample_first_rows:?}"
            ),
        }
    }
}

impl std::error::Error for QuotientMetalError {}
