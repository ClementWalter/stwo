//! Per-proof injection point for an optional whole composition-polynomial stage.
//!
//! A stage is scoped to one `prove` call. It may decline before submitting work by
//! returning `Ok(None)`; the prover then executes its unchanged host implementation.
//! Once device work has been submitted, failure is terminal and must be returned as
//! `Err`—falling back after submission could consume mutated inputs or outputs and
//! would make backend-participation telemetry misleading.

use thiserror::Error;

use super::air::component_prover::{ComponentProvers, Trace};
use super::backend::Backend;
use super::poly::circle::SecureCirclePoly;
use super::poly::twiddles::TwiddleTree;
use super::EvaluationMode;
use crate::core::fields::qm31::SecureField;

/// Immutable inputs supplied to an optional composition-polynomial stage.
///
/// The stage receives the already-drawn random coefficient. Consulting a stage does
/// not draw from or otherwise mutate the Fiat–Shamir transcript.
pub struct CompositionPolynomialStageInputs<'call, 'components, 'trace, B: Backend> {
    pub component_provers: &'call ComponentProvers<'components, B>,
    pub random_coeff: SecureField,
    pub trace: &'call Trace<'trace, B>,
    pub twiddles: &'call TwiddleTree<B>,
    pub log_blowup_factor: u32,
    pub composition_log_degree_bound: u32,
    pub total_constraints: usize,
    pub evaluation_mode: EvaluationMode,
}

/// Terminal failure after an optional composition stage submitted device work.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum CompositionPolynomialStageError {
    #[error("the submitted composition-polynomial stage failed")]
    SubmittedComputationFailed,
    #[error("the submitted composition-polynomial stage returned an invalid result")]
    InvalidResult,
}

/// Optional, per-proof producer of the complete composition polynomial.
///
/// Implementations must admit their full shape and resources before submission. An
/// unsupported shape or unavailable device is `Ok(None)`. `Ok(Some(_))` must contain
/// the complete polynomial, including any work that the implementation did not place
/// on its accelerator. An error after submission is terminal.
pub trait CompositionPolynomialStage<B: Backend> {
    fn try_compute(
        &self,
        inputs: CompositionPolynomialStageInputs<'_, '_, '_, B>,
    ) -> Result<Option<SecureCirclePoly<B>>, CompositionPolynomialStageError>;
}

pub(crate) fn validate_stage_result<B: Backend>(
    polynomial: SecureCirclePoly<B>,
    expected_log_size: u32,
) -> Result<SecureCirclePoly<B>, CompositionPolynomialStageError> {
    if polynomial.log_size() != expected_log_size {
        return Err(CompositionPolynomialStageError::InvalidResult);
    }
    Ok(polynomial)
}

#[cfg(test)]
mod tests {
    use super::{validate_stage_result, CompositionPolynomialStageError};
    use crate::core::fields::m31::BaseField;
    use crate::prover::backend::CpuBackend;
    use crate::prover::poly::circle::{CircleCoefficients, SecureCirclePoly};

    #[test]
    fn submitted_failures_are_explicit_terminal_errors() {
        assert_eq!(
            CompositionPolynomialStageError::SubmittedComputationFailed.to_string(),
            "the submitted composition-polynomial stage failed"
        );
    }

    #[test]
    fn wrong_size_stage_polynomial_is_rejected() {
        let polynomial = SecureCirclePoly::<CpuBackend>(std::array::from_fn(|_| {
            CircleCoefficients::new(vec![BaseField::default(); 8])
        }));
        assert!(matches!(
            validate_stage_result(polynomial, 4),
            Err(CompositionPolynomialStageError::InvalidResult)
        ));
    }
}
