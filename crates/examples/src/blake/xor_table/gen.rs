#[macro_export]
macro_rules! xor_table_gen {
    ($modname:tt, $elements:tt, $elem_bits:literal, $expand_bits:literal) => {
        pub struct XorTableLookupData<const ELEM_BITS: u32, const EXPAND_BITS: u32> {
            pub xor_accum: XorAccumulator<ELEM_BITS, EXPAND_BITS>,
        }

        pub fn generate_trace<const ELEM_BITS: u32, const EXPAND_BITS: u32>(
            xor_accum: XorAccumulator<ELEM_BITS, EXPAND_BITS>,
        ) -> (
            ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
            XorTableLookupData<ELEM_BITS, EXPAND_BITS>,
        ) {
            (
                xor_accum
                    .mults
                    .iter()
                    .map(|mult| {
                        CircleEvaluation::new(
                            CanonicCoset::new(
                                XorTable::new(ELEM_BITS, EXPAND_BITS, 0).column_bits(),
                            )
                            .circle_domain(),
                            mult.clone(),
                        )
                    })
                    .collect_vec(),
                XorTableLookupData { xor_accum },
            )
        }

        /// Generates the interaction trace for the xor table.
        /// Returns the interaction trace, the Preprocessed trace, and the claimed sum.
        pub fn generate_interaction_trace<
            const ELEM_BITS: u32,
            const EXPAND_BITS: u32,
            X: Relation<PackedBaseField, PackedSecureField> + Sync,
        >(
            lookup_data: XorTableLookupData<ELEM_BITS, EXPAND_BITS>,
            lookup_elements: &X,
        ) -> (
            ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
            SecureField,
        ) {
            let limb_bits = XorTable::new(ELEM_BITS, EXPAND_BITS, 0).limb_bits();
            let _span = span!(Level::INFO, "Xor interaction trace").entered();
            let offsets_vec = u32x16::from_array(std::array::from_fn(|i| i as u32));
            let mut logup_gen =
                LogupTraceGenerator::new(XorTable::new(ELEM_BITS, EXPAND_BITS, 0).column_bits());

            // Batch pairs of multiplicity columns into one logup column each; an odd
            // final column gets its own. There are 2^(2*EXPAND_BITS) columns, one per
            // combination of (ah, bh). Each column has 2^(2*LIMB_BITS) rows, packed in
            // N_LANES.
            let mults = &lookup_data.xor_accum.mults;
            let n_pairs = mults.len() / 2;
            let n_cols = n_pairs + mults.len() % 2;

            // Extract ah, bh from a column index.
            let split_idx = |i: u32| (i >> EXPAND_BITS, i & ((1 << EXPAND_BITS) - 1));
            // The lookup tuple (a, b, a ^ b) for multiplicity column `i` at `vec_row`.
            // vec_row is LIMB_BITS of al and LIMB_BITS - LOG_N_LANES of bl; bll is the
            // consecutive numbers 0..N_LANES-1.
            let lookup_at = |i: u32, vec_row: u32| {
                let (ah, bh) = split_idx(i);
                let al = vec_row >> (limb_bits - LOG_N_LANES);
                let blh = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);
                let a = u32x16::splat((ah << limb_bits) | al);
                let b = u32x16::splat((bh << limb_bits) | (blh << LOG_N_LANES)) | offsets_vec;
                let c = a ^ b;
                let p: PackedSecureField = lookup_elements.combine(
                    &[a, b, c].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }),
                );
                p
            };

            logup_gen.cols_from_fn(n_cols, |col, vec_row| {
                if col < n_pairs {
                    let (i0, i1) = ((2 * col) as u32, (2 * col + 1) as u32);
                    let p0 = lookup_at(i0, vec_row as u32);
                    let p1 = lookup_at(i1, vec_row as u32);
                    let num = p1 * mults[i0 as usize].data[vec_row]
                        + p0 * mults[i1 as usize].data[vec_row];
                    (-num, p0 * p1)
                } else {
                    // The odd final column.
                    let i = (mults.len() - 1) as u32;
                    let p = lookup_at(i, vec_row as u32);
                    let num = mults[i as usize].data[vec_row];
                    (PackedSecureField::from(-num), p)
                }
            });

            logup_gen.finalize_last()
        }
    };
}
pub(crate) use xor_table_gen;
