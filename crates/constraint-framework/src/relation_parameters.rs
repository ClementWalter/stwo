//! Read-only extraction of the runtime parameters carried by AIR relations.
//!
//! This evaluator runs an existing [`FrameworkEval`](crate::FrameworkEval) with
//! unit masks and ignores constraints. Each observed relation is queried through
//! its public [`Relation::combine`](crate::Relation::combine) contract, so generated
//! relation wrappers do not need to expose their internal lookup elements.

use std::collections::BTreeMap;

use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

use crate::{EvalAtRow, Relation, RelationEntry};

/// Runtime `z` and `alpha_i` values for one named relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationParameterSet {
    pub z: SecureField,
    pub alphas: Vec<SecureField>,
}

/// Evaluator that recovers every relation parameter referenced by an AIR.
#[derive(Clone, Debug, Default)]
pub struct RelationParameterEvaluator {
    parameters: BTreeMap<String, RelationParameterSet>,
}

impl RelationParameterEvaluator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parameters(&self) -> &BTreeMap<String, RelationParameterSet> {
        &self.parameters
    }

    pub fn into_parameters(self) -> BTreeMap<String, RelationParameterSet> {
        self.parameters
    }
}

impl EvalAtRow for RelationParameterEvaluator {
    type F = BaseField;
    type EF = SecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        _interaction: usize,
        _offsets: [isize; N],
    ) -> [Self::F; N] {
        [BaseField::one(); N]
    }

    fn add_constraint<G>(&mut self, _constraint: G)
    where
        Self::EF: core::ops::Mul<G, Output = Self::EF> + From<G>,
    {
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        SecureField::from_m31_array(values)
    }

    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    ) {
        let arity = entry.relation.get_size();
        let mut basis = vec![BaseField::zero(); arity];
        let z = -entry.relation.combine(&basis);
        let alphas = (0..arity)
            .map(|index| {
                basis[index] = BaseField::one();
                let alpha = entry.relation.combine(&basis) + z;
                basis[index] = BaseField::zero();
                alpha
            })
            .collect();
        let recovered = RelationParameterSet { z, alphas };
        if let Some(previous) = self
            .parameters
            .insert(entry.relation.get_name().to_owned(), recovered.clone())
        {
            assert_eq!(
                previous,
                recovered,
                "relation parameters changed within one AIR evaluation for {}",
                entry.relation.get_name(),
            );
        }
    }

    fn write_logup_frac(&mut self, _fraction: stwo::core::Fraction<Self::EF, Self::EF>) {}

    fn finalize_logup_batched(&mut self, _batch_size: usize) {}

    fn finalize_logup(&mut self) {}

    fn finalize_logup_in_pairs(&mut self) {}
}

#[cfg(test)]
mod tests {
    use num_traits::{One, Zero};
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;

    use super::RelationParameterEvaluator;
    use crate::{EvalAtRow, Relation, RelationEntry};

    crate::relation!(TestRelation, 3);

    #[test]
    fn recovers_relation_parameters_through_combine_contract() {
        let relation = TestRelation::dummy();
        let values = [BaseField::one(); 2];
        let mut evaluator = RelationParameterEvaluator::new();
        evaluator.add_to_relation(RelationEntry::new(&relation, SecureField::one(), &values));

        let recovered = &evaluator.parameters()["TestRelation"];
        assert_eq!(recovered.alphas.len(), 3);
        let zero_combination: SecureField = relation.combine(&[BaseField::zero(); 3]);
        assert_eq!(zero_combination, -recovered.z);
        for index in 0..3 {
            let mut basis = [BaseField::zero(); 3];
            basis[index] = BaseField::one();
            let unit_combination: SecureField = relation.combine(&basis);
            assert_eq!(unit_combination, recovered.alphas[index] - recovered.z);
        }
    }
}
