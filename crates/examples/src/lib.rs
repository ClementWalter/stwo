#![feature(portable_simd, iter_array_chunks, array_chunks)]
pub mod air_shape;
pub mod blake;
pub mod plonk;
pub mod poseidon;
pub mod state_machine;
pub mod wide_fibonacci;
pub mod xor;

/// Prints a hash of the proof's debug representation when `PROOF_HASH` is set, for
/// cross-build bit-identity checks (the debug form covers every field of the proof).
pub fn maybe_dump_proof_hash<T: std::fmt::Debug>(label: &str, proof: &T) {
    if std::env::var("PROOF_HASH").is_ok() {
        use std::hash::{Hash, Hasher};
        let repr = std::format!("{proof:?}");
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        repr.hash(&mut hasher);
        std::println!(
            "PROOF_HASH[{label}]={:016x} len={}",
            hasher.finish(),
            repr.len()
        );
    }
}
