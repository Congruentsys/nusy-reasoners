//! # nusy-reasoner — the Reasoner contract (VY-D / EX-4746)
//!
//! The single trait every reasoner in the NuSy family implements (V18-GENERALIZED-REASONING
//! §3). One contract over the shared Y-graph:
//!
//! - **Query in** — a goal claim + the relevant graph slice.
//! - **Out** — an answer **plus a proof/derivation trace, a provability tag, and provenance**.
//! - **Self-description** — its [`CompetenceEnvelope`] (what queries it covers), its
//!   [`Substrate`] (neural / symbolic / mixed), and its [`Guarantee`] (sound? complete?).
//!
//! The **proof/provenance trace is the universal currency** — every reasoner, even a neural
//! one, emits *why* in a form the gate and other reasoners can consume. That single invariant
//! is what lets reasoners compose, lets the gate certify provability, and lets the brain audit.
//!
//! ## The guarantee invariant (the load-bearing property)
//!
//! **A heuristic answer can never construct a `Proven` tag.** [`Provability`] is *computed from
//! the proof trace* ([`Answer::provability`]), never set by the reasoner. Only a **complete
//! [`ProofTrace::Derivation`]** yields [`Provability::Proven`]; a neural [`ProofTrace::Evidence`]
//! is *always* [`Provability::Heuristic`]. A neural reasoner holds no derivation tree, so it is
//! *structurally* unable to mint `Proven` — provability flows from evidence, not assertion.
//!
//! ## Arrow-free by design
//!
//! This crate depends only on [`nusy_unify`] (pure term algebra). It does **not** depend on
//! arrow-core or the engine — the engine's `ProofTree` (EX-4592) *adapts into* [`DerivationTrace`]
//! from the outside. That keeps the contract movable to the public `nusy-reasoners` FOSS suite
//! (VY-C). Object-safe ([`Reasoner`] is usable as `dyn Reasoner` / `Box<dyn Reasoner>` by the
//! router).
//!
//! ```
//! use nusy_reasoner::*;
//! use nusy_unify::Triple;
//!
//! // A symbolic reasoner answers WITH a complete derivation → Proven.
//! let proven = Answer {
//!     value: Some(Triple::new("p1", "at_risk", "fall")),
//!     proof: ProofTrace::Derivation(DerivationTrace::Derived {
//!         conclusion: Triple::new("p1", "at_risk", "fall"),
//!         rule_id: "at-risk-fall".into(),
//!         premises: vec![DerivationTrace::Axiom(Triple::new("p1", "has_condition", "osteoporosis"))],
//!     }),
//!     provenance: vec!["chunk-7".into()],
//! };
//! assert_eq!(proven.provability(), Provability::Proven);
//!
//! // A neural reasoner answers WITH evidence → Heuristic, never Proven.
//! let neural = Answer {
//!     value: Some(Triple::new("p1", "at_risk", "stroke")),
//!     proof: ProofTrace::Evidence { confidence: 0.82, why: vec!["age + bp pattern".into()] },
//!     provenance: vec![],
//! };
//! assert_eq!(neural.provability(), Provability::Heuristic);
//! ```

pub mod compose;
pub use compose::{Pipeline, provability_min};

use nusy_unify::Triple;

/// What a reasoner runs on: a goal claim plus the relevant graph slice (facts/rules/ontology),
/// substrate-neutral so any reasoner family can consume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    /// The claim/goal to answer.
    pub goal: Triple,
    /// The relevant slice of the Y-graph (seed facts, applicable rules as triples, terminology).
    pub context: Vec<Triple>,
}

impl Query {
    /// A query for `goal` with no extra context.
    pub fn new(goal: Triple) -> Self {
        Self {
            goal,
            context: Vec::new(),
        }
    }
}

/// A reasoner's answer: the value, its proof trace, and provenance. **Provability is derived
/// from the proof, never stored** ([`Answer::provability`]) — that is the guarantee invariant.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The derived fact / answer; `None` when the reasoner abstained.
    pub value: Option<Triple>,
    /// The universal currency — *why* this answer holds (or that it is unprovable).
    pub proof: ProofTrace,
    /// Source attribution (chunk ids, citations, the rule chain's seed provenance).
    pub provenance: Vec<String>,
}

impl Answer {
    /// The reasoner abstained — no value, no proof.
    pub fn abstained() -> Self {
        Self {
            value: None,
            proof: ProofTrace::None,
            provenance: Vec::new(),
        }
    }

    /// **Computed**, not set: `Proven` only when the trace is a complete derivation; a neural
    /// `Evidence` trace is always `Heuristic`; no trace is `Abstained`. The reasoner cannot lie.
    pub fn provability(&self) -> Provability {
        self.proof.provability()
    }
}

/// Three-valued provability — **computed from the [`ProofTrace`], never asserted by a reasoner.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provability {
    /// Answered with a **complete sound derivation** — certifiable, zero-hallucination.
    Proven,
    /// Best-effort / neural — evidence + calibrated confidence, explicitly **not** a proof.
    Heuristic,
    /// No answer within the reasoner's guarantee.
    Abstained,
}

/// The universal proof currency. A symbolic reasoner emits a [`Derivation`](ProofTrace::Derivation);
/// a neural one emits [`Evidence`](ProofTrace::Evidence) (explicitly unprovable); abstention emits
/// [`None`](ProofTrace::None).
#[derive(Debug, Clone, PartialEq)]
pub enum ProofTrace {
    /// A symbolic derivation tree. The engine's `ProofTree` (EX-4592) adapts *into* this from
    /// outside the crate. Only a **complete** derivation (every leaf an axiom) is `Proven`.
    Derivation(DerivationTrace),
    /// Neural / best-effort: a calibrated confidence and the evidence behind it — **never a proof**.
    Evidence {
        /// Calibrated confidence in `[0, 1]`.
        confidence: f64,
        /// Human/queryable evidence the answer rests on.
        why: Vec<String>,
    },
    /// No trace — the reasoner abstained.
    None,
}

impl ProofTrace {
    /// **The guarantee invariant.** `Proven` iff a *complete* derivation; `Evidence` (neural) is
    /// *always* `Heuristic` and can never become `Proven`; `None` is `Abstained`. An incomplete
    /// derivation (a dangling/unproven premise) degrades to `Heuristic`, never `Proven`.
    pub fn provability(&self) -> Provability {
        match self {
            ProofTrace::Derivation(d) if d.is_complete() => Provability::Proven,
            ProofTrace::Derivation(_) => Provability::Heuristic,
            ProofTrace::Evidence { .. } => Provability::Heuristic,
            ProofTrace::None => Provability::Abstained,
        }
    }
}

/// A substrate-neutral derivation: a conclusion derived by a rule from its premises, bottoming
/// out at seed axioms (leaves). The engine's proof tree adapts into this — the trait crate never
/// depends on the engine (keeps it arrow-free / FOSS-movable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DerivationTrace {
    /// A seed fact — an asserted premise, not derived (a leaf).
    Axiom(Triple),
    /// A derived conclusion: the rule that fired and the proofs of its premises.
    Derived {
        /// The fact this step concludes.
        conclusion: Triple,
        /// The rule that produced it.
        rule_id: String,
        /// Sub-proofs of the premises that satisfied the rule body.
        premises: Vec<DerivationTrace>,
    },
}

impl DerivationTrace {
    /// **Complete** iff every branch bottoms out at an [`Axiom`](DerivationTrace::Axiom) leaf —
    /// i.e. no premise is left unproven. A [`Derived`](DerivationTrace::Derived) node with **no
    /// premises** is an *unsupported* step (a rule that fired on nothing) and is therefore **not**
    /// complete. Completeness is what distinguishes a `Proven` answer from a partial one.
    pub fn is_complete(&self) -> bool {
        match self {
            DerivationTrace::Axiom(_) => true,
            DerivationTrace::Derived { premises, .. } => {
                !premises.is_empty() && premises.iter().all(DerivationTrace::is_complete)
            }
        }
    }

    /// Proof depth: an axiom is 0; a derivation is 1 + its deepest premise.
    pub fn depth(&self) -> usize {
        match self {
            DerivationTrace::Axiom(_) => 0,
            DerivationTrace::Derived { premises, .. } => {
                1 + premises
                    .iter()
                    .map(DerivationTrace::depth)
                    .max()
                    .unwrap_or(0)
            }
        }
    }

    /// Every rule id used anywhere in this tree (deepest-first, with repeats).
    pub fn rule_ids(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_rule_ids(&mut out);
        out
    }
    fn collect_rule_ids<'a>(&'a self, out: &mut Vec<&'a str>) {
        if let DerivationTrace::Derived {
            rule_id, premises, ..
        } = self
        {
            for p in premises {
                p.collect_rule_ids(out);
            }
            out.push(rule_id);
        }
    }
}

/// The substrate a reasoner runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substrate {
    /// Pure symbolic (forward-chain, gate, do-calculus) — can be `Proven`.
    Symbolic,
    /// Neural (an LLM behind the gate) — `Heuristic` only, never authoritative.
    Neural,
    /// Mixed: symbolic spine with neural proposals (proven where it can, heuristic elsewhere).
    Mixed,
}

/// A reasoner's guarantee about its answers — what the router checks against the required bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Guarantee {
    /// Never asserts a false claim (a `Proven` answer is always correct).
    pub sound: bool,
    /// Finds every answer within its competence (no false abstention).
    pub complete: bool,
    /// Answers carry calibrated probabilities rather than certainties.
    pub probabilistic: bool,
}

/// A query shape a reasoner soundly covers — e.g. ground-pattern deduction, do-calculus
/// intervention. **Data, not an enum**, so the family is open: a new reasoner declares its
/// competence without changing this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryShape {
    /// Human-readable name of the shape.
    pub name: String,
    /// Predicates this shape covers. **Empty = any predicate** (a wildcard shape).
    pub predicates: Vec<String>,
}

impl QueryShape {
    /// Does this shape cover `query`'s goal predicate?
    pub fn covers(&self, query: &Query) -> bool {
        self.predicates.is_empty() || self.predicates.contains(&query.goal.predicate)
    }
}

/// The set of query shapes a reasoner soundly covers — its competence, as data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompetenceEnvelope {
    /// The covered shapes.
    pub shapes: Vec<QueryShape>,
}

impl CompetenceEnvelope {
    /// Does **any** shape in the envelope cover `query`? (The router uses this to choose a
    /// reasoner whose competence covers the claim before checking its guarantee meets the bar.)
    pub fn covers(&self, query: &Query) -> bool {
        self.shapes.iter().any(|s| s.covers(query))
    }
}

/// **The contract every reasoner implements.** Object-safe so the reasoner-router can hold a
/// heterogeneous `Vec<Box<dyn Reasoner>>` and pick by competence + guarantee.
///
/// The router's logic (VY-D E2+): *given a claim and a required provability level, route to the
/// reasoner(s) whose [`competence_envelope`](Reasoner::competence_envelope) covers it and whose
/// [`guarantee`](Reasoner::guarantee) meets the bar; return the answer with its proof and a
/// provability certificate; abstain/flag when no reasoner meets the bar.*
pub trait Reasoner {
    /// Answer `query`: derive-with-proof, or carry evidence, or abstain. The returned
    /// [`Answer`]'s [`provability`](Answer::provability) is computed from its proof — a reasoner
    /// cannot claim `Proven` it did not derive.
    fn answer(&self, query: &Query) -> Answer;

    /// What kinds of query this reasoner soundly covers.
    fn competence_envelope(&self) -> &CompetenceEnvelope;

    /// Neural / symbolic / mixed.
    fn substrate(&self) -> Substrate;

    /// Sound? complete? probabilistic?
    fn guarantee(&self) -> Guarantee;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    /// A toy symbolic reasoner: answers `at_risk(p,fall)` with a complete derivation.
    struct SymbolicFallRisk {
        envelope: CompetenceEnvelope,
    }
    impl SymbolicFallRisk {
        fn new() -> Self {
            Self {
                envelope: CompetenceEnvelope {
                    shapes: vec![QueryShape {
                        name: "fall-risk deduction".into(),
                        predicates: vec!["at_risk".into()],
                    }],
                },
            }
        }
    }
    impl Reasoner for SymbolicFallRisk {
        fn answer(&self, query: &Query) -> Answer {
            if !self.competence_envelope().covers(query) {
                return Answer::abstained();
            }
            let concl = query.goal.clone();
            Answer {
                value: Some(concl.clone()),
                proof: ProofTrace::Derivation(DerivationTrace::Derived {
                    conclusion: concl,
                    rule_id: "at-risk-fall".into(),
                    premises: vec![DerivationTrace::Axiom(t(
                        &query.goal.subject,
                        "has_condition",
                        "osteoporosis",
                    ))],
                }),
                provenance: vec!["seed:chunk-7".into()],
            }
        }
        fn competence_envelope(&self) -> &CompetenceEnvelope {
            &self.envelope
        }
        fn substrate(&self) -> Substrate {
            Substrate::Symbolic
        }
        fn guarantee(&self) -> Guarantee {
            Guarantee {
                sound: true,
                complete: true,
                probabilistic: false,
            }
        }
    }

    /// A toy neural reasoner: answers anything, but only with evidence (never a proof).
    struct NeuralGuesser;
    impl Reasoner for NeuralGuesser {
        fn answer(&self, query: &Query) -> Answer {
            Answer {
                value: Some(query.goal.clone()),
                proof: ProofTrace::Evidence {
                    confidence: 0.7,
                    why: vec!["pattern match".into()],
                },
                provenance: vec![],
            }
        }
        fn competence_envelope(&self) -> &CompetenceEnvelope {
            // Wildcard: covers any predicate.
            static ANY: std::sync::OnceLock<CompetenceEnvelope> = std::sync::OnceLock::new();
            ANY.get_or_init(|| CompetenceEnvelope {
                shapes: vec![QueryShape {
                    name: "any".into(),
                    predicates: vec![],
                }],
            })
        }
        fn substrate(&self) -> Substrate {
            Substrate::Neural
        }
        fn guarantee(&self) -> Guarantee {
            Guarantee {
                sound: false,
                complete: false,
                probabilistic: true,
            }
        }
    }

    #[test]
    fn symbolic_complete_derivation_is_proven() {
        let r = SymbolicFallRisk::new();
        let a = r.answer(&Query::new(t("p1", "at_risk", "fall")));
        assert_eq!(a.provability(), Provability::Proven);
        assert!(a.value.is_some());
        assert!(!a.provenance.is_empty());
    }

    #[test]
    fn guarantee_invariant_neural_is_never_proven() {
        // THE load-bearing test: a neural/heuristic answer can NEVER be Proven.
        let r = NeuralGuesser;
        let a = r.answer(&Query::new(t("p1", "at_risk", "stroke")));
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_ne!(a.provability(), Provability::Proven);
        // Structural: an Evidence trace's provability is Heuristic regardless of confidence.
        let high = ProofTrace::Evidence {
            confidence: 0.999,
            why: vec![],
        };
        assert_eq!(high.provability(), Provability::Heuristic);
    }

    #[test]
    fn incomplete_derivation_degrades_to_heuristic_not_proven() {
        // A derivation with an UNSUPPORTED premise (a Derived node that fired on nothing) is not
        // a closed proof → Heuristic, never Proven. This is the partial-proof boundary.
        let incomplete = DerivationTrace::Derived {
            conclusion: t("p1", "at_risk", "fall"),
            rule_id: "at-risk-fall".into(),
            premises: vec![DerivationTrace::Derived {
                conclusion: t("p1", "has_condition", "osteoporosis"),
                rule_id: "unsupported".into(),
                premises: vec![], // dangling: a rule that fired on no premises — not closed
            }],
        };
        assert!(!incomplete.is_complete());
        assert_eq!(
            ProofTrace::Derivation(incomplete).provability(),
            Provability::Heuristic
        );
    }

    #[test]
    fn abstention_is_abstained() {
        let a = Answer::abstained();
        assert_eq!(a.provability(), Provability::Abstained);
        assert!(a.value.is_none());
    }

    #[test]
    fn trait_is_object_safe_router_can_hold_dyn() {
        // The router holds a heterogeneous family behind `dyn Reasoner`.
        let family: Vec<Box<dyn Reasoner>> =
            vec![Box::new(SymbolicFallRisk::new()), Box::new(NeuralGuesser)];
        let q = Query::new(t("p1", "at_risk", "fall"));
        // Pick the first reasoner whose envelope covers the claim (toy router).
        let chosen = family
            .iter()
            .find(|r| r.competence_envelope().covers(&q))
            .expect("a reasoner covers it");
        // The symbolic one is registered first and covers `at_risk` → Proven.
        assert_eq!(chosen.answer(&q).provability(), Provability::Proven);
    }

    #[test]
    fn derivation_trace_depth_and_rule_ids() {
        let d = DerivationTrace::Derived {
            conclusion: t("a", "p", "c"),
            rule_id: "r-top".into(),
            premises: vec![DerivationTrace::Derived {
                conclusion: t("a", "p", "b"),
                rule_id: "r-mid".into(),
                premises: vec![DerivationTrace::Axiom(t("a", "seed", "b"))],
            }],
        };
        assert_eq!(d.depth(), 2);
        assert!(d.is_complete());
        assert_eq!(d.rule_ids(), vec!["r-mid", "r-top"]);
    }

    #[test]
    fn competence_envelope_covers_by_predicate() {
        let env = CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "bp".into(),
                predicates: vec!["bp_target".into()],
            }],
        };
        assert!(env.covers(&Query::new(t("p1", "bp_target", "140/90"))));
        assert!(!env.covers(&Query::new(t("p1", "at_risk", "fall"))));
    }
}
