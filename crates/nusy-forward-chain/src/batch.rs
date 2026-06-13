//! Arrow substrate for the reasoning engine (EX-4668, VY-4667 phase 1).
//!
//! Defines the engine-facing Arrow representation of ground triples and derivations:
//! dictionary-encoded `subject`/`predicate`/`object` columns ([`triple_batch_schema`]),
//! a [`TripleBatch`] view that accumulates per-round [`RecordBatch`]es (the **delta seam**
//! the incremental engine EX-4593 will match against), a parallel [`DerivationBatch`]
//! (conclusion + rule id + premise **row refs**), and lossless `Vec<Triple>` ↔ batch
//! converters.
//!
//! ## Placement
//!
//! This lives in `nusy-forward-chain` (not `nusy-arrow-core`): the converters need
//! [`nusy_unify::Triple`], and the foundational `nusy-arrow-core` substrate must not
//! depend on a V18 engine crate. `nusy-forward-chain` already sits above `nusy-unify`,
//! so the dependency points the right way and no cycle is possible.
//!
//! ## Scope (phase 1)
//!
//! Substrate only — no matching, no fixpoint, no indexes. [`TripleBatch::position_of`]
//! is a linear scan: it exists to support lossless conversion (premise row refs), not as
//! an engine lookup path. The Arrow matcher (EX-4669) and the Arrow fixpoint with real
//! indexing (EX-4670) build on top of this module; the public Vec-based engine API is
//! untouched.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ListBuilder, StringDictionaryBuilder, UInt32Array, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, UInt32Type};
use arrow::record_batch::RecordBatch;
use nusy_unify::Triple;

use crate::Derivation;

/// The three term positions of a triple column set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripleCol {
    /// The `subject` column.
    Subject,
    /// The `predicate` column.
    Predicate,
    /// The `object` column.
    Object,
}

impl TripleCol {
    fn index(self) -> usize {
        match self {
            TripleCol::Subject => 0,
            TripleCol::Predicate => 1,
            TripleCol::Object => 2,
        }
    }
}

/// Errors from batch construction / decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// A derivation premise is not a row of the fact batch — row refs cannot be resolved.
    PremiseNotInFacts {
        /// The premise that has no fact row.
        premise: Triple,
    },
    /// A stored row reference points past the end of the fact batch.
    RowOutOfRange {
        /// The offending row reference.
        row: u64,
        /// Total rows in the fact batch.
        len: u64,
    },
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchError::PremiseNotInFacts { premise } => write!(
                f,
                "derivation premise {} {} {} is not a row of the fact batch",
                premise.subject, premise.predicate, premise.object
            ),
            BatchError::RowOutOfRange { row, len } => {
                write!(
                    f,
                    "row reference {row} out of range (fact batch has {len} rows)"
                )
            }
        }
    }
}

impl std::error::Error for BatchError {}

/// Dictionary-encoded term column type: `Dictionary<UInt32, Utf8>`.
fn term_type() -> DataType {
    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8))
}

/// Schema for a batch of ground triples: dictionary-encoded `subject`, `predicate`,
/// `object` (all non-null). Dictionary encoding gives term-id (`u32`) access for joins
/// and amortizes repeated terms — graph data repeats terms heavily.
pub fn triple_batch_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("subject", term_type(), false),
        Field::new("predicate", term_type(), false),
        Field::new("object", term_type(), false),
    ]))
}

/// Schema for a batch of derivations, parallel to a fact [`TripleBatch`]:
/// the conclusion's terms (dictionary-encoded), the firing rule's id, and the premises
/// as a list of **row references** (`UInt64`) into the fact batch's global row space.
/// Premises are always existing facts in a saturation, so row refs are total.
pub fn derivation_batch_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("conclusion_subject", term_type(), false),
        Field::new("conclusion_predicate", term_type(), false),
        Field::new("conclusion_object", term_type(), false),
        Field::new("rule_id", term_type(), false),
        Field::new(
            "premise_rows",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            false,
        ),
    ]))
}

/// Build one triple [`RecordBatch`] from a slice of triples (order and duplicates
/// preserved). An empty slice yields an empty batch with the canonical schema.
fn triples_to_record_batch(triples: &[Triple]) -> RecordBatch {
    let mut subjects = StringDictionaryBuilder::<UInt32Type>::new();
    let mut predicates = StringDictionaryBuilder::<UInt32Type>::new();
    let mut objects = StringDictionaryBuilder::<UInt32Type>::new();
    for t in triples {
        subjects.append_value(&t.subject);
        predicates.append_value(&t.predicate);
        objects.append_value(&t.object);
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(subjects.finish()),
        Arc::new(predicates.finish()),
        Arc::new(objects.finish()),
    ];
    RecordBatch::try_new(triple_batch_schema(), columns)
        .expect("triple columns match the canonical schema by construction")
}

/// Zero-copy `&str` for one cell of a dictionary-encoded term column.
fn dict_value(batch: &RecordBatch, col: usize, row: usize) -> &str {
    let dict = batch
        .column(col)
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<UInt32Type>>()
        .expect("term column is Dictionary<UInt32, Utf8> by schema");
    let key = dict.keys().value(row) as usize;
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("dictionary values are Utf8 by schema");
    values.value(key)
}

/// A growing set of ground triples as Arrow batches, one [`RecordBatch`] per **round**.
///
/// Rows have a stable global index (append-only): row `i` is the `i`-th triple across
/// rounds in append order. Each `append_triples` call adds one round — the engine's
/// fixpoint loop appends each round's newly derived facts as its own batch, which is
/// exactly the per-round **delta** the semi-naive evaluator (EX-4593) matches against.
#[derive(Debug, Clone, Default)]
pub struct TripleBatch {
    rounds: Vec<RecordBatch>,
    /// Cumulative row count at the end of each round (parallel to `rounds`).
    offsets: Vec<usize>,
}

impl TripleBatch {
    /// An empty batch (no rounds).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a single-round batch from triples (order and duplicates preserved).
    pub fn from_triples(triples: &[Triple]) -> Self {
        let mut tb = Self::new();
        tb.append_triples(triples);
        tb
    }

    /// Append one round of triples as a new [`RecordBatch`]. Appending an empty slice
    /// is a no-op (no empty round is recorded — a fixpoint round that derives nothing
    /// terminates the loop rather than growing the store).
    pub fn append_triples(&mut self, triples: &[Triple]) {
        if triples.is_empty() {
            return;
        }
        let new_total = self.len() + triples.len();
        self.rounds.push(triples_to_record_batch(triples));
        self.offsets.push(new_total);
    }

    /// The per-round record batches, in append order — the delta seam for EX-4593.
    pub fn rounds(&self) -> &[RecordBatch] {
        &self.rounds
    }

    /// Total triples across all rounds.
    pub fn len(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// Is the batch empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Locate a global row: `(round index, row index within round)`.
    fn locate(&self, row: usize) -> (usize, usize) {
        assert!(
            row < self.len(),
            "row {row} out of range ({} rows)",
            self.len()
        );
        // First round whose cumulative end exceeds `row`.
        let round = self.offsets.partition_point(|&end| end <= row);
        let start = if round == 0 {
            0
        } else {
            self.offsets[round - 1]
        };
        (round, row - start)
    }

    /// Zero-copy term access: the `&str` at (`row`, `col`), borrowed from the underlying
    /// Arrow buffer (no allocation).
    pub fn term_at(&self, row: usize, col: TripleCol) -> &str {
        let (round, local) = self.locate(row);
        dict_value(&self.rounds[round], col.index(), local)
    }

    /// The dictionary key (term id) at (`row`, `col`). Term ids are **per-round**
    /// dictionary keys — stable within a round's batch, not across rounds.
    pub fn term_id_at(&self, row: usize, col: TripleCol) -> u32 {
        let (round, local) = self.locate(row);
        let dict = self.rounds[round]
            .column(col.index())
            .as_any()
            .downcast_ref::<arrow::array::DictionaryArray<UInt32Type>>()
            .expect("term column is Dictionary<UInt32, Utf8> by schema");
        dict.keys().value(local)
    }

    /// Materialize the triple at a global row.
    pub fn triple_at(&self, row: usize) -> Triple {
        Triple::new(
            self.term_at(row, TripleCol::Subject),
            self.term_at(row, TripleCol::Predicate),
            self.term_at(row, TripleCol::Object),
        )
    }

    /// Global row of the **first** occurrence of `t`, if present. Linear scan —
    /// converter support (premise row resolution), not an engine lookup path
    /// (EX-4670 maintains real indexes).
    pub fn position_of(&self, t: &Triple) -> Option<u64> {
        (0..self.len())
            .find(|&row| {
                self.term_at(row, TripleCol::Subject) == t.subject
                    && self.term_at(row, TripleCol::Predicate) == t.predicate
                    && self.term_at(row, TripleCol::Object) == t.object
            })
            .map(|row| row as u64)
    }

    /// Iterate all triples in global row order (materializing each).
    pub fn iter(&self) -> impl Iterator<Item = Triple> + '_ {
        (0..self.len()).map(|row| self.triple_at(row))
    }

    /// Materialize the whole batch back to a `Vec<Triple>` (exact round-trip of
    /// [`TripleBatch::from_triples`] / [`TripleBatch::append_triples`] input order).
    pub fn to_triples(&self) -> Vec<Triple> {
        self.iter().collect()
    }
}

/// A batch of derivations parallel to a fact [`TripleBatch`]: conclusion terms, rule id,
/// and premises as row refs into the fact batch ([`derivation_batch_schema`]).
#[derive(Debug, Clone)]
pub struct DerivationBatch {
    batch: RecordBatch,
}

impl DerivationBatch {
    /// Encode derivations against `facts`, resolving each premise to its first fact row.
    ///
    /// Errors with [`BatchError::PremiseNotInFacts`] if any premise is not a row of
    /// `facts` — in a well-formed saturation every premise is an existing fact.
    pub fn from_derivations(
        derivations: &[Derivation],
        facts: &TripleBatch,
    ) -> Result<Self, BatchError> {
        // One pass over `facts` builds the first-occurrence row index; per-premise
        // `position_of` scans would be O(derivations × premises × facts).
        let mut row_of: HashMap<Triple, u64> = HashMap::new();
        for row in 0..facts.len() {
            row_of.entry(facts.triple_at(row)).or_insert(row as u64);
        }

        let mut conc_s = StringDictionaryBuilder::<UInt32Type>::new();
        let mut conc_p = StringDictionaryBuilder::<UInt32Type>::new();
        let mut conc_o = StringDictionaryBuilder::<UInt32Type>::new();
        let mut rule_ids = StringDictionaryBuilder::<UInt32Type>::new();
        let mut premise_rows = ListBuilder::new(UInt64Builder::new());

        for d in derivations {
            conc_s.append_value(&d.conclusion.subject);
            conc_p.append_value(&d.conclusion.predicate);
            conc_o.append_value(&d.conclusion.object);
            rule_ids.append_value(&d.rule_id);
            for premise in &d.premises {
                let row =
                    row_of
                        .get(premise)
                        .copied()
                        .ok_or_else(|| BatchError::PremiseNotInFacts {
                            premise: premise.clone(),
                        })?;
                premise_rows.values().append_value(row);
            }
            premise_rows.append(true);
        }

        let columns: Vec<ArrayRef> = vec![
            Arc::new(conc_s.finish()),
            Arc::new(conc_p.finish()),
            Arc::new(conc_o.finish()),
            Arc::new(rule_ids.finish()),
            Arc::new(premise_rows.finish()),
        ];
        let batch = RecordBatch::try_new(derivation_batch_schema(), columns)
            .expect("derivation columns match the canonical schema by construction");
        Ok(Self { batch })
    }

    /// The underlying [`RecordBatch`].
    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Number of derivations.
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    /// Is the batch empty?
    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }

    /// The conclusion triple of derivation row `i` (no premise resolution).
    pub fn conclusion_at(&self, i: usize) -> Triple {
        Triple::new(
            dict_value(&self.batch, 0, i),
            dict_value(&self.batch, 1, i),
            dict_value(&self.batch, 2, i),
        )
    }

    /// Decode derivation row `i`, resolving premise row refs through `facts`.
    ///
    /// Errors with [`BatchError::RowOutOfRange`] if a row ref does not resolve —
    /// i.e. `facts` is not the batch this was encoded against.
    pub fn decode_row(&self, i: usize, facts: &TripleBatch) -> Result<Derivation, BatchError> {
        let premise_lists = self
            .batch
            .column(4)
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .expect("premise_rows is a List<UInt64> by schema");
        let rows = premise_lists.value(i);
        let rows = rows
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("premise_rows items are UInt64 by schema");
        let mut premises = Vec::with_capacity(rows.len());
        for j in 0..rows.len() {
            let row = rows.value(j);
            if row as usize >= facts.len() {
                return Err(BatchError::RowOutOfRange {
                    row,
                    len: facts.len() as u64,
                });
            }
            premises.push(facts.triple_at(row as usize));
        }
        Ok(Derivation {
            conclusion: self.conclusion_at(i),
            rule_id: dict_value(&self.batch, 3, i).to_string(),
            premises,
        })
    }

    /// Decode back to [`Derivation`]s, resolving premise row refs through `facts`.
    ///
    /// Errors with [`BatchError::RowOutOfRange`] if a row ref does not resolve —
    /// i.e. `facts` is not the batch this was encoded against.
    pub fn to_derivations(&self, facts: &TripleBatch) -> Result<Vec<Derivation>, BatchError> {
        (0..self.batch.num_rows())
            .map(|i| self.decode_row(i, facts))
            .collect()
    }
}

/// Keys of a round's term column as a typed [`UInt32Array`] view (zero-copy).
/// Exposed for the Arrow matcher (EX-4669) to join on term ids within a round.
pub fn round_term_ids(batch: &RecordBatch, col: TripleCol) -> &UInt32Array {
    batch
        .column(col.index())
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<UInt32Type>>()
        .expect("term column is Dictionary<UInt32, Utf8> by schema")
        .keys()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    #[test]
    fn schema_uses_dictionary_encoded_terms() {
        let schema = triple_batch_schema();
        for field in schema.fields() {
            assert_eq!(field.data_type(), &term_type(), "{}", field.name());
            assert!(!field.is_nullable());
        }
    }

    #[test]
    fn empty_round_trips() {
        let tb = TripleBatch::from_triples(&[]);
        assert!(tb.is_empty());
        assert_eq!(tb.len(), 0);
        assert!(tb.rounds().is_empty());
        assert_eq!(tb.to_triples(), Vec::<Triple>::new());
    }

    #[test]
    fn order_and_duplicates_round_trip() {
        let triples = vec![
            t("a", "parent", "b"),
            t("b", "parent", "c"),
            t("a", "parent", "b"), // duplicate preserved
            t("c", "likes", "a"),
        ];
        let tb = TripleBatch::from_triples(&triples);
        assert_eq!(tb.len(), 4);
        assert_eq!(tb.to_triples(), triples);
    }

    #[test]
    fn unicode_terms_round_trip() {
        let triples = vec![
            t("patientë", "hâs_condition", "östeoporosis"),
            t("変数", "関係", "値"),
        ];
        let tb = TripleBatch::from_triples(&triples);
        assert_eq!(tb.to_triples(), triples);
        assert_eq!(tb.term_at(1, TripleCol::Subject), "変数");
    }

    #[test]
    fn append_rounds_concatenate_and_expose_deltas() {
        let round1 = vec![t("a", "p", "b"), t("b", "p", "c")];
        let round2 = vec![t("a", "anc", "b")];
        let mut tb = TripleBatch::from_triples(&round1);
        tb.append_triples(&round2);
        tb.append_triples(&[]); // no-op
        assert_eq!(tb.rounds().len(), 2);
        assert_eq!(tb.rounds()[1].num_rows(), 1);
        let mut all = round1.clone();
        all.extend(round2.clone());
        assert_eq!(tb.to_triples(), all);
        assert_eq!(tb.triple_at(2), round2[0]);
    }

    #[test]
    fn term_at_and_term_ids_are_consistent() {
        let triples = vec![t("a", "p", "b"), t("a", "q", "b"), t("c", "p", "a")];
        let tb = TripleBatch::from_triples(&triples);
        // Same term → same per-round id; different term → different id.
        assert_eq!(
            tb.term_id_at(0, TripleCol::Subject),
            tb.term_id_at(1, TripleCol::Subject)
        );
        assert_ne!(
            tb.term_id_at(0, TripleCol::Subject),
            tb.term_id_at(2, TripleCol::Subject)
        );
        assert_eq!(tb.term_at(2, TripleCol::Object), "a");
    }

    #[test]
    fn position_of_finds_first_occurrence_only() {
        let triples = vec![t("a", "p", "b"), t("x", "y", "z"), t("a", "p", "b")];
        let tb = TripleBatch::from_triples(&triples);
        assert_eq!(tb.position_of(&t("a", "p", "b")), Some(0));
        assert_eq!(tb.position_of(&t("x", "y", "z")), Some(1));
        assert_eq!(tb.position_of(&t("nope", "p", "b")), None);
    }

    #[test]
    fn derivations_round_trip_against_fact_batch() {
        let facts = vec![
            t("a", "parent", "b"),
            t("b", "parent", "c"),
            t("a", "grandparent", "c"),
        ];
        let tb = TripleBatch::from_triples(&facts);
        let derivs = vec![Derivation {
            conclusion: t("a", "grandparent", "c"),
            rule_id: "gp".to_string(),
            premises: vec![t("a", "parent", "b"), t("b", "parent", "c")],
        }];
        let db = DerivationBatch::from_derivations(&derivs, &tb).unwrap();
        assert_eq!(db.len(), 1);
        assert_eq!(db.to_derivations(&tb).unwrap(), derivs);
    }

    #[test]
    fn derivation_with_unknown_premise_errors() {
        let tb = TripleBatch::from_triples(&[t("a", "p", "b")]);
        let derivs = vec![Derivation {
            conclusion: t("a", "q", "b"),
            rule_id: "r".to_string(),
            premises: vec![t("missing", "p", "b")],
        }];
        let err = DerivationBatch::from_derivations(&derivs, &tb).unwrap_err();
        assert!(matches!(err, BatchError::PremiseNotInFacts { .. }));
    }

    #[test]
    fn decoding_against_wrong_facts_errors_out_of_range() {
        let full = TripleBatch::from_triples(&[t("a", "p", "b"), t("b", "p", "c")]);
        let derivs = vec![Derivation {
            conclusion: t("x", "q", "y"),
            rule_id: "r".to_string(),
            premises: vec![t("b", "p", "c")], // row 1 in `full`
        }];
        let db = DerivationBatch::from_derivations(&derivs, &full).unwrap();
        let truncated = TripleBatch::from_triples(&[t("a", "p", "b")]);
        let err = db.to_derivations(&truncated).unwrap_err();
        assert!(matches!(err, BatchError::RowOutOfRange { row: 1, len: 1 }));
    }
}
