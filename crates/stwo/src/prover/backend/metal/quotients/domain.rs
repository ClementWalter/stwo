use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::fields::m31::BaseField;
use crate::core::poly::circle::CircleDomain;
use crate::core::utils::bit_reverse_index;

pub(super) struct XyColumns {
    pub(super) xs: Vec<BaseField>,
    pub(super) ys: Vec<BaseField>,
}

/// Returns the domain's bit-reversed coordinate columns, computing each immutable
/// `(size, shift)` table once and sharing it across proofs.
pub(super) fn get(domain: CircleDomain) -> Arc<XyColumns> {
    type Cache = HashMap<(u32, usize), Arc<XyColumns>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

    let key = (domain.log_size(), domain.half_coset.initial_index.0);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(columns) = cache.lock().unwrap().get(&key) {
        return columns.clone();
    }
    let log_size = domain.log_size();
    let mut xs = vec![BaseField::from_u32_unchecked(0); domain.size()];
    let mut ys = vec![BaseField::from_u32_unchecked(0); domain.size()];
    let fill = |(row, (x, y)): (usize, (&mut BaseField, &mut BaseField))| {
        let point = domain.at(bit_reverse_index(row, log_size));
        *x = point.x;
        *y = point.y;
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        xs.par_iter_mut()
            .zip(ys.par_iter_mut())
            .enumerate()
            .for_each(fill);
    }
    #[cfg(not(feature = "parallel"))]
    xs.iter_mut().zip(ys.iter_mut()).enumerate().for_each(fill);
    let columns = Arc::new(XyColumns { xs, ys });
    cache.lock().unwrap().entry(key).or_insert(columns).clone()
}
