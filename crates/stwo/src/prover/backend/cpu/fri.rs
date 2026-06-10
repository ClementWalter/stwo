#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::CpuBackend;
use crate::core::circle::Coset;
use crate::core::fft::ibutterfly;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::poly::line::LineDomain;
use crate::core::utils::bit_reverse_index;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

impl FriOps for CpuBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alphas: &[SecureField],
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let fold_step = alphas.len();
        assert!(fold_step >= 1);

        let mut res = fold_line_cpu(eval, alphas[0]);
        for &alpha in &alphas[1..] {
            res = fold_line_cpu(&res, alpha);
        }
        res
    }

    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        fold_circle_into_line_cpu(src, alpha)
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        let lambda = Self::decomposition_coefficient(eval);
        let mut g_values = unsafe { SecureColumnByCoords::<Self>::uninitialized(eval.len()) };

        let domain_size = eval.len();
        let half_domain_size = domain_size / 2;

        for i in 0..half_domain_size {
            let x = eval.values.at(i);
            let val = x - lambda;
            g_values.set(i, val);
        }
        for i in half_domain_size..domain_size {
            let x = eval.values.at(i);
            let val = x + lambda;
            g_values.set(i, val);
        }

        let g = SecureEvaluation::new(eval.domain, g_values);
        (g, lambda)
    }
}

/// TODO: Almost duplicate code of [`crate::core::fri::fold_line`]. Consider refactor.
pub fn fold_line_cpu(
    eval: &LineEvaluation<CpuBackend>,
    alpha: SecureField,
) -> LineEvaluation<CpuBackend> {
    let n = eval.len();
    assert!(n >= 2, "Evaluation too small");

    let domain = eval.domain();
    let half_n = n / 2;
    let folded_values = unsafe { SecureColumnByCoords::<CpuBackend>::uninitialized(half_n) };
    let mut folded_values = LineEvaluation::new(domain.double(), folded_values);

    // Pairs are independent; fold disjoint chunks concurrently, batching each chunk's
    // domain-point inversions into a single field inversion.
    let chunk_size = 1 << 10;
    let mut chunk_views = {
        let [c0, c1, c2, c3]: &mut [Vec<BaseField>; 4] = &mut folded_values.values.columns;
        (c0.chunks_mut(chunk_size))
            .zip(c1.chunks_mut(chunk_size))
            .zip(c2.chunks_mut(chunk_size))
            .zip(c3.chunks_mut(chunk_size))
            .enumerate()
            .map(|(i, (((d0, d1), d2), d3))| (i * chunk_size, [d0, d1, d2, d3]))
            .collect::<Vec<_>>()
    };

    let process_chunk = |(start, chunk): &mut (usize, [&mut [BaseField]; 4])| {
        let start = *start;
        let rows = chunk[0].len();
        let xs: Vec<BaseField> = (0..rows)
            .map(|idx| domain.at(bit_reverse_index((start + idx) << 1, domain.log_size())))
            .collect();
        let x_invs = BaseField::batch_inverse(&xs);
        for (idx, x_inv) in x_invs.into_iter().enumerate() {
            let i = start + idx;
            let f_x = eval.values.at(i << 1);
            let f_neg_x = eval.values.at((i << 1) + 1);
            let (mut f0, mut f1) = (f_x, f_neg_x);
            ibutterfly(&mut f0, &mut f1, x_inv);
            let [v0, v1, v2, v3] = (f0 + alpha * f1).to_m31_array();
            chunk[0][idx] = v0;
            chunk[1][idx] = v1;
            chunk[2][idx] = v2;
            chunk[3][idx] = v3;
        }
    };

    #[cfg(feature = "parallel")]
    chunk_views.par_iter_mut().for_each(process_chunk);
    #[cfg(not(feature = "parallel"))]
    chunk_views.iter_mut().for_each(process_chunk);

    folded_values
}

/// TODO: Almost duplicate code of [`crate::core::fri::fold_circle_into_line`]. Consider refactor.
pub fn fold_circle_into_line_cpu(
    src: &SecureEvaluation<CpuBackend, BitReversedOrder>,
    alpha: SecureField,
) -> LineEvaluation<CpuBackend> {
    let domain = src.domain;
    let line_log_size = src.domain.log_size() - 1;
    let dst_domain = LineDomain::new(Coset::half_odds(line_log_size));
    let values = unsafe { SecureColumnByCoords::uninitialized(1 << line_log_size) };
    let mut dst = LineEvaluation::new(dst_domain, values);

    // Pairs are independent; fold disjoint chunks concurrently, batching each chunk's
    // domain-point inversions into a single field inversion.
    let chunk_size = 1 << 10;
    let mut chunk_views = {
        let [c0, c1, c2, c3]: &mut [Vec<BaseField>; 4] = &mut dst.values.columns;
        (c0.chunks_mut(chunk_size))
            .zip(c1.chunks_mut(chunk_size))
            .zip(c2.chunks_mut(chunk_size))
            .zip(c3.chunks_mut(chunk_size))
            .enumerate()
            .map(|(i, (((d0, d1), d2), d3))| (i * chunk_size, [d0, d1, d2, d3]))
            .collect::<Vec<_>>()
    };

    let process_chunk = |(start, chunk): &mut (usize, [&mut [BaseField]; 4])| {
        let start = *start;
        let rows = chunk[0].len();
        let ys: Vec<BaseField> = (0..rows)
            .map(|idx| {
                domain
                    .at(bit_reverse_index((start + idx) << 1, domain.log_size()))
                    .y
            })
            .collect();
        let y_invs = BaseField::batch_inverse(&ys);
        for (idx, y_inv) in y_invs.into_iter().enumerate() {
            let i = start + idx;
            let f_p = src.values.at(i << 1);
            let f_neg_p = src.values.at((i << 1) + 1);
            // Calculate `f0(px)` and `f1(px)` such that `2f(p) = f0(px) + py * f1(px)`.
            let (mut f0_px, mut f1_px) = (f_p, f_neg_p);
            ibutterfly(&mut f0_px, &mut f1_px, y_inv);
            let f_prime = alpha * f1_px + f0_px;
            let [v0, v1, v2, v3] = f_prime.to_m31_array();
            chunk[0][idx] = v0;
            chunk[1][idx] = v1;
            chunk[2][idx] = v2;
            chunk[3][idx] = v3;
        }
    };

    #[cfg(feature = "parallel")]
    chunk_views.par_iter_mut().for_each(process_chunk);
    #[cfg(not(feature = "parallel"))]
    chunk_views.iter_mut().for_each(process_chunk);

    dst
}

impl CpuBackend {
    /// Used to decompose a general polynomial to a polynomial inside the fft-space, and
    /// the remainder terms.
    /// A coset-diff on a [`CircleCoefficients`] that is in the FFT space will return zero.
    ///
    /// Let N be the domain size, Let h be a coset size N/2. Using lemma #7 from the CircleStark
    /// paper, <f,V_h> = lambda<V_h,V_h> = lambda\*N => lambda = f(0)\*V_h(0) + f(1)*V_h(1) + .. +
    /// f(N-1)\*V_h(N-1). The Vanishing polynomial of a cannonic coset sized half the circle
    /// domain,evaluated on the circle domain, is [(1, -1, -1, 1)] repeating. This becomes
    /// alternating [+-1] in our NaturalOrder, and [(+, +, +, ... , -, -)] in bit reverse.
    /// Explicitly, lambda\*N = sum(+f(0..N/2)) + sum(-f(N/2..)).
    ///
    /// # Warning
    /// This function assumes the blowupfactor is 2
    ///
    /// [`CircleCoefficients`]: crate::core::poly::circle::CircleCoefficients
    fn decomposition_coefficient(eval: &SecureEvaluation<Self, BitReversedOrder>) -> SecureField {
        let domain_size = 1 << eval.domain.log_size();
        let half_domain_size = domain_size / 2;

        // eval is in bit-reverse, hence all the positive factors are in the first half, opposite to
        // the latter.
        let a_sum = (0..half_domain_size)
            .map(|i| eval.values.at(i))
            .sum::<SecureField>();
        let b_sum = (half_domain_size..domain_size)
            .map(|i| eval.values.at(i))
            .sum::<SecureField>();

        // lambda = sum(+-f(p)) / 2N.
        (a_sum - b_sum) / BaseField::from_u32_unchecked(domain_size as u32)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::m31;
    use crate::prover::backend::cpu::{CpuCircleEvaluation, CpuCirclePoly};
    use crate::prover::backend::CpuBackend;
    use crate::prover::fri::FriOps;
    use crate::prover::poly::circle::SecureEvaluation;
    use crate::prover::poly::BitReversedOrder;
    use crate::prover::secure_column::SecureColumnByCoords;

    #[test]
    fn decompose_coeff_out_fft_space_test() {
        for domain_log_size in 5..12 {
            let domain_log_half_size = domain_log_size - 1;
            let s = CanonicCoset::new(domain_log_size);
            let domain = s.circle_domain();

            let mut coeffs = vec![BaseField::zero(); 1 << domain_log_size];

            // Polynomial is out of FFT space.
            coeffs[1 << domain_log_half_size] = m31!(1);
            assert!(!CpuCirclePoly::new(coeffs.clone()).is_in_fft_space(domain_log_half_size));

            let poly = CpuCirclePoly::new(coeffs);
            let values = poly.evaluate(domain);
            let secure_column = SecureColumnByCoords {
                columns: [
                    values.values.clone(),
                    values.values.clone(),
                    values.values.clone(),
                    values.values.clone(),
                ],
            };
            let secure_eval = SecureEvaluation::<CpuBackend, BitReversedOrder>::new(
                domain,
                secure_column.clone(),
            );

            let (g, lambda) = CpuBackend::decompose(&secure_eval);

            // Sanity check.
            assert_ne!(lambda, SecureField::zero());

            // Assert the new polynomial is in the FFT space.
            for i in 0..4 {
                let basefield_column = g.columns[i].clone();
                let eval = CpuCircleEvaluation::new(domain, basefield_column);
                let coeffs = eval.interpolate().coeffs;
                assert!(CpuCirclePoly::new(coeffs).is_in_fft_space(domain_log_half_size));
            }
        }
    }
}
