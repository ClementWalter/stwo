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

            let n_vec_rows: usize =
                1 << (XorTable::new(ELEM_BITS, EXPAND_BITS, 0).column_bits() - LOG_N_LANES);

            // Iterate each pair of columns, to batch their lookup together.
            // There are 2^(2*EXPAND_BITS) column, for each combination of ah, bh.
            let (pairs, rem) = lookup_data.xor_accum.mults.as_chunks::<2>();
            for (pair_idx, [mults0, mults1]) in pairs.iter().enumerate() {
                let (i0, i1) = (2 * pair_idx, 2 * pair_idx + 1);

                // Extract ah, bh from column index.
                let ah0 = i0 as u32 >> EXPAND_BITS;
                let bh0 = i0 as u32 & ((1 << EXPAND_BITS) - 1);
                let ah1 = i1 as u32 >> EXPAND_BITS;
                let bh1 = i1 as u32 & ((1 << EXPAND_BITS) - 1);

                // Each column has 2^(2*LIMB_BITS) rows, packed in N_LANES.
                let frac_at_row = |vec_row: usize| {
                    let vec_row = vec_row as u32;
                    // vec_row is LIMB_BITS of al and LIMB_BITS - LOG_N_LANES of bl.
                    // Extract al, blh from vec_row.
                    let al = vec_row >> (limb_bits - LOG_N_LANES);
                    let blh = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);

                    // Construct the 3 vectors a, b, c.
                    let a0 = u32x16::splat((ah0 << limb_bits) | al);
                    let a1 = u32x16::splat((ah1 << limb_bits) | al);
                    // bll is just the consecutive numbers 0 .. N_LANES-1.
                    let b0 = u32x16::splat((bh0 << limb_bits) | (blh << LOG_N_LANES)) | offsets_vec;
                    let b1 = u32x16::splat((bh1 << limb_bits) | (blh << LOG_N_LANES)) | offsets_vec;

                    let c0 = a0 ^ b0;
                    let c1 = a1 ^ b1;

                    let p0: PackedSecureField = lookup_elements.combine(
                        &[a0, b0, c0].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }),
                    );
                    let p1: PackedSecureField = lookup_elements.combine(
                        &[a1, b1, c1].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }),
                    );

                    let num =
                        p1 * mults0.data[vec_row as usize] + p0 * mults1.data[vec_row as usize];
                    let denom = p0 * p1;
                    (-num, denom)
                };

                #[cfg(feature = "parallel")]
                logup_gen.col_from_par_iter((0..n_vec_rows).into_par_iter().map(frac_at_row));
                #[cfg(not(feature = "parallel"))]
                logup_gen.col_from_iter((0..n_vec_rows).map(frac_at_row));
            }

            // If there is an odd number of lookup expressions, handle the last one.
            if let Some(mults) = rem.last() {
                let i = lookup_data.xor_accum.mults.len() - 1;
                let ah = i as u32 >> EXPAND_BITS;
                let bh = i as u32 & ((1 << EXPAND_BITS) - 1);

                let frac_at_row = |vec_row: usize| {
                    let vec_row = vec_row as u32;
                    // vec_row is LIMB_BITS of a, and LIMB_BITS - LOG_N_LANES of b.
                    let al = vec_row >> (limb_bits - LOG_N_LANES);
                    let a = u32x16::splat((ah << limb_bits) | al);
                    let bm = vec_row & ((1 << (limb_bits - LOG_N_LANES)) - 1);
                    let b = u32x16::splat((bh << limb_bits) | (bm << LOG_N_LANES)) | offsets_vec;

                    let c = a ^ b;

                    let p: PackedSecureField = lookup_elements.combine(
                        &[a, b, c].map(|x| unsafe { PackedBaseField::from_simd_unchecked(x) }),
                    );

                    let num = mults.data[vec_row as usize];
                    (PackedSecureField::from(-num), p)
                };

                #[cfg(feature = "parallel")]
                logup_gen.col_from_par_iter((0..n_vec_rows).into_par_iter().map(frac_at_row));
                #[cfg(not(feature = "parallel"))]
                logup_gen.col_from_iter((0..n_vec_rows).map(frac_at_row));
            }

            logup_gen.finalize_last()
        }
    };
}
pub(crate) use xor_table_gen;
