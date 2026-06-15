//! # nusy-inductive — the inductive reasoner (VY-B E1, EX-4887)
//!
//! The LRM's *learning* half: given a set of observed instances, **induce candidate rules** that
//! generalize them — "entities with attribute *body* tend to also have attribute *head*" — scored
//! by **support** (how many entities exhibit the body) and **confidence** (how often the head holds
//! among them). This is the engine behind scientific evidence synthesis and rule discovery.
//!
//! ## The guarantee invariant — induction is HEURISTIC, never Proven
//!
//! A generalization is **not** a proof: "every swan I have seen is white" does not *prove* the next
//! swan is white. So every answer this reasoner produces carries a [`ProofTrace::Evidence`] trace,
//! whose [`provability`](nusy_reasoner::Answer::provability) is *always*
//! [`Provability::Heuristic`](nusy_reasoner::Provability::Heuristic). The induced engine holds no
//! [`DerivationTrace`](nusy_reasoner::DerivationTrace), so it is **structurally unable to mint
//! `Proven`** — exactly as the Reasoner contract requires. The induced rules feed downstream
//! reasoning as *candidates*; only a sound deductive engine (with a complete derivation) can ever
//! certify them `Proven`.
//!
//! ## Generic by construction
//!
//! The engine contains **no domain literals** — every predicate/object it generalizes over comes
//! from the instance data, never from the code. Induce over clinical facts, legal facts, or animal
//! facts with the same engine.
//!
//! ```
//! use nusy_inductive::{InductionConfig, InductiveReasoner};
//! use nusy_reasoner::{Provability, Query, Reasoner};
//! use nusy_unify::Triple;
//!
//! // Observed: three birds, all of which fly.
//! let instances = vec![
//!     Triple::new("tweety", "is_a", "bird"),  Triple::new("tweety", "can", "fly"),
//!     Triple::new("robin",  "is_a", "bird"),  Triple::new("robin",  "can", "fly"),
//!     Triple::new("crow",   "is_a", "bird"),  Triple::new("crow",   "can", "fly"),
//! ];
//! let r = InductiveReasoner::from_instances(&instances, &InductionConfig::default());
//!
//! // A new bird: induction proposes it can fly — but only HEURISTICALLY.
//! let q = Query {
//!     goal: Triple::new("sparrow", "can", "fly"),
//!     context: vec![Triple::new("sparrow", "is_a", "bird")],
//! };
//! let a = r.answer(&q);
//! assert_eq!(a.provability(), Provability::Heuristic); // a generalization is not a proof
//! assert_ne!(a.provability(), Provability::Proven);
//! ```

use std::collections::{BTreeMap, BTreeSet};

use nusy_reasoner::{
    Answer, CompetenceEnvelope, Guarantee, ProofTrace, Query, QueryShape, Reasoner, Substrate,
};
use nusy_unify::Triple;

/// An attribute of an entity: a `(predicate, object)` pair. The unit a rule generalizes over —
/// e.g. `("is_a", "bird")` or `("can", "fly")`.
pub type Attr = (String, String);

/// A candidate generalization induced from instances: *entities with `body` tend to have `head`*.
///
/// Scored, not asserted: `confidence = positives / support`. **A `CandidateRule` is a hypothesis,
/// not a theorem** — applying it yields a [`Provability::Heuristic`](nusy_reasoner::Provability)
/// answer.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRule {
    /// Antecedent attribute — the rule fires for entities that have this.
    pub body: Attr,
    /// Consequent attribute — what the rule generalizes the entity to also have.
    pub head: Attr,
    /// Number of entities exhibiting `body` (the rule's reach).
    pub support: usize,
    /// Number of entities exhibiting **both** `body` and `head`.
    pub positives: usize,
    /// `positives / support` ∈ (0, 1] — how reliable the generalization is in the data.
    pub confidence: f64,
}

impl CandidateRule {
    /// Render the rule for an evidence trace, e.g. `is_a=bird ⇒ can=fly`.
    fn describe(&self) -> String {
        format!(
            "{}={} ⇒ {}={} (support={}, confidence={:.3})",
            self.body.0, self.body.1, self.head.0, self.head.1, self.support, self.confidence
        )
    }
}

/// Thresholds for [`induce`]. A candidate is kept only if it clears **both** — enough evidence
/// (`support`) and enough reliability (`confidence`).
#[derive(Debug, Clone, Copy)]
pub struct InductionConfig {
    /// Minimum entities exhibiting the body — guards against generalizing from a single example.
    pub min_support: usize,
    /// Minimum `positives / support` — guards against generalizing through counterexamples.
    pub min_confidence: f64,
}

impl Default for InductionConfig {
    /// Conservative defaults: at least 2 supporting entities and a majority (> 0.5) confidence.
    fn default() -> Self {
        Self {
            min_support: 2,
            min_confidence: 0.5,
        }
    }
}

/// Group instances by entity (subject) → its set of `(predicate, object)` attributes.
fn entities_with_attrs(instances: &[Triple]) -> BTreeMap<String, BTreeSet<Attr>> {
    let mut map: BTreeMap<String, BTreeSet<Attr>> = BTreeMap::new();
    for t in instances {
        map.entry(t.subject.clone())
            .or_default()
            .insert((t.predicate.clone(), t.object.clone()));
    }
    map
}

/// **Induce candidate rules from instances.** For every ordered pair of distinct attributes
/// `(body, head)` observed in the data, measure support (entities with `body`) and confidence
/// (`P(head | body)`); keep those clearing [`InductionConfig`]. Deterministic: the result is sorted
/// by confidence ↓, support ↓, then lexicographically, so the same instances always induce the same
/// rules.
pub fn induce(instances: &[Triple], cfg: &InductionConfig) -> Vec<CandidateRule> {
    let entities = entities_with_attrs(instances);

    // All attributes observed anywhere (sorted → deterministic enumeration).
    let mut all_attrs: BTreeSet<Attr> = BTreeSet::new();
    for attrs in entities.values() {
        all_attrs.extend(attrs.iter().cloned());
    }

    let mut rules = Vec::new();
    for body in &all_attrs {
        // support = entities exhibiting the body.
        let body_entities: Vec<&BTreeSet<Attr>> = entities
            .values()
            .filter(|attrs| attrs.contains(body))
            .collect();
        let support = body_entities.len();
        if support < cfg.min_support {
            continue;
        }
        for head in &all_attrs {
            if head == body {
                continue; // a rule must relate two *distinct* attributes
            }
            let positives = body_entities
                .iter()
                .filter(|attrs| attrs.contains(head))
                .count();
            if positives == 0 {
                continue;
            }
            let confidence = positives as f64 / support as f64;
            if confidence + f64::EPSILON < cfg.min_confidence {
                continue;
            }
            rules.push(CandidateRule {
                body: body.clone(),
                head: head.clone(),
                support,
                positives,
                confidence,
            });
        }
    }

    // Deterministic ordering: strongest first, ties broken lexicographically.
    rules.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.support.cmp(&a.support))
            .then((&a.body, &a.head).cmp(&(&b.body, &b.head)))
    });
    rules
}

/// An [`Reasoner`] that answers from **induced** rules. Every answer it gives is
/// [`Provability::Heuristic`](nusy_reasoner::Provability) — induction generalizes, it does not
/// prove.
pub struct InductiveReasoner {
    rules: Vec<CandidateRule>,
    envelope: CompetenceEnvelope,
}

impl InductiveReasoner {
    /// Induce rules from `instances` and build a reasoner whose competence covers exactly the
    /// head predicates it learned.
    pub fn from_instances(instances: &[Triple], cfg: &InductionConfig) -> Self {
        Self::from_rules(induce(instances, cfg))
    }

    /// Build directly from already-induced rules (e.g. a curated rule set).
    pub fn from_rules(rules: Vec<CandidateRule>) -> Self {
        let predicates: BTreeSet<String> = rules.iter().map(|r| r.head.0.clone()).collect();
        let envelope = CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "induced generalization".into(),
                predicates: predicates.into_iter().collect(),
            }],
        };
        Self { rules, envelope }
    }

    /// The induced rules (strongest first).
    pub fn rules(&self) -> &[CandidateRule] {
        &self.rules
    }

    /// The strongest induced rule whose head matches `goal` and whose body is present in `known`.
    fn matching_rule(&self, goal: &Triple, known: &BTreeSet<Attr>) -> Option<&CandidateRule> {
        let head: Attr = (goal.predicate.clone(), goal.object.clone());
        // `rules` is sorted strongest-first, so the first match is the most confident.
        self.rules
            .iter()
            .find(|r| r.head == head && known.contains(&r.body))
    }
}

impl Reasoner for InductiveReasoner {
    /// Fire the strongest induced rule whose body is satisfied by the query's known context, and
    /// return the head **as evidence** (never a derivation). Abstains when no induced rule covers
    /// the goal or the entity lacks the body attribute.
    fn answer(&self, query: &Query) -> Answer {
        let known: BTreeSet<Attr> = query
            .context
            .iter()
            .filter(|t| t.subject == query.goal.subject)
            .map(|t| (t.predicate.clone(), t.object.clone()))
            .collect();

        match self.matching_rule(&query.goal, &known) {
            Some(rule) => Answer {
                value: Some(query.goal.clone()),
                // Evidence — NEVER a Derivation. This is what keeps induced answers Heuristic.
                proof: ProofTrace::Evidence {
                    confidence: rule.confidence,
                    why: vec![format!("induced: {}", rule.describe())],
                },
                provenance: vec![format!(
                    "induced-rule:{}={}->{}={}",
                    rule.body.0, rule.body.1, rule.head.0, rule.head.1
                )],
            },
            None => Answer::abstained(),
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    /// `Substrate::Symbolic`: induction is an *algorithm* over triples (no neural net) — matching
    /// the `nusy-abduction` precedent for a symbolic-but-non-proof process. Its non-provability is
    /// carried by the `Evidence` proof + `sound: false`, **not** by the substrate tag.
    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }

    /// **Unsound and incomplete by nature** — a generalization can be wrong (unsound) and the data
    /// may not cover every true rule (incomplete); answers carry a confidence (probabilistic). This
    /// is exactly why its answers can never be `Proven`.
    fn guarantee(&self) -> Guarantee {
        Guarantee {
            sound: false,
            complete: false,
            probabilistic: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_reasoner::Provability;

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    /// A query with a known-context graph slice (the contract has no builder; `context` is public).
    fn q(goal: Triple, context: Vec<Triple>) -> Query {
        Query { goal, context }
    }

    /// Four birds, all fly → induce `is_a=bird ⇒ can=fly` at confidence 1.0.
    fn bird_instances() -> Vec<Triple> {
        vec![
            t("tweety", "is_a", "bird"),
            t("tweety", "can", "fly"),
            t("robin", "is_a", "bird"),
            t("robin", "can", "fly"),
            t("crow", "is_a", "bird"),
            t("crow", "can", "fly"),
            t("eagle", "is_a", "bird"),
            t("eagle", "can", "fly"),
        ]
    }

    #[test]
    fn induces_high_confidence_rule() {
        let rules = induce(&bird_instances(), &InductionConfig::default());
        let bird_fly = rules
            .iter()
            .find(|r| {
                r.body == ("is_a".into(), "bird".into()) && r.head == ("can".into(), "fly".into())
            })
            .expect("should induce is_a=bird ⇒ can=fly");
        assert_eq!(bird_fly.support, 4);
        assert_eq!(bird_fly.positives, 4);
        assert!((bird_fly.confidence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn confidence_reflects_counterexamples() {
        // 4 birds, only 3 fly (penguin doesn't) → confidence 3/4 = 0.75.
        let mut inst = vec![
            t("tweety", "is_a", "bird"),
            t("tweety", "can", "fly"),
            t("robin", "is_a", "bird"),
            t("robin", "can", "fly"),
            t("crow", "is_a", "bird"),
            t("crow", "can", "fly"),
            t("pengu", "is_a", "bird"), // a bird that does NOT fly
        ];
        inst.push(t("pengu", "can", "swim"));
        let rules = induce(&inst, &InductionConfig::default());
        let bird_fly = rules
            .iter()
            .find(|r| {
                r.body == ("is_a".into(), "bird".into()) && r.head == ("can".into(), "fly".into())
            })
            .expect("rule still clears 0.5 confidence");
        assert_eq!(bird_fly.support, 4);
        assert_eq!(bird_fly.positives, 3);
        assert!((bird_fly.confidence - 0.75).abs() < 1e-9);
    }

    #[test]
    fn respects_min_confidence() {
        // Half the birds fly → confidence 0.5; require 0.8 → rule dropped.
        let inst = vec![
            t("a", "is_a", "bird"),
            t("a", "can", "fly"),
            t("b", "is_a", "bird"),
            t("b", "can", "fly"),
            t("c", "is_a", "bird"),
            t("d", "is_a", "bird"),
        ];
        let cfg = InductionConfig {
            min_support: 2,
            min_confidence: 0.8,
        };
        let rules = induce(&inst, &cfg);
        assert!(
            !rules.iter().any(|r| r.head == ("can".into(), "fly".into())),
            "0.5-confidence rule must be filtered by min_confidence 0.8"
        );
    }

    #[test]
    fn respects_min_support() {
        // Only one entity exhibits the body → below min_support 2 → no rule.
        let inst = vec![t("lonely", "is_a", "unicorn"), t("lonely", "can", "fly")];
        let rules = induce(&inst, &InductionConfig::default());
        assert!(
            rules.is_empty(),
            "a single example must not induce a rule (min_support)"
        );
    }

    #[test]
    fn induction_is_deterministic() {
        let a = induce(&bird_instances(), &InductionConfig::default());
        let b = induce(&bird_instances(), &InductionConfig::default());
        assert_eq!(
            a, b,
            "same instances must induce the same rules in the same order"
        );
    }

    #[test]
    fn no_domain_literals_generic_over_legal_facts() {
        // The same engine generalizes a totally different domain — proves there are no hardcoded
        // predicates/objects in the engine.
        let inst = vec![
            t("contract_a", "has_clause", "arbitration"),
            t("contract_a", "is", "enforceable"),
            t("contract_b", "has_clause", "arbitration"),
            t("contract_b", "is", "enforceable"),
            t("contract_c", "has_clause", "arbitration"),
            t("contract_c", "is", "enforceable"),
        ];
        let rules = induce(&inst, &InductionConfig::default());
        assert!(
            rules
                .iter()
                .any(|r| r.body == ("has_clause".into(), "arbitration".into())
                    && r.head == ("is".into(), "enforceable".into())),
            "engine must induce over arbitrary domains with no code changes"
        );
    }

    // ── The guarantee invariant (load-bearing) ──────────────────────────────

    #[test]
    fn induced_answer_is_heuristic_never_proven() {
        let r = InductiveReasoner::from_instances(&bird_instances(), &InductionConfig::default());
        let q = q(
            t("sparrow", "can", "fly"),
            vec![t("sparrow", "is_a", "bird")],
        );
        let a = r.answer(&q);
        assert_eq!(a.value, Some(t("sparrow", "can", "fly")));
        // THE invariant: a generalization is never a proof.
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_ne!(a.provability(), Provability::Proven);
        // And it is honestly self-described as unsound.
        assert!(!r.guarantee().sound);
    }

    #[test]
    fn abstains_when_entity_lacks_the_body() {
        let r = InductiveReasoner::from_instances(&bird_instances(), &InductionConfig::default());
        // No context saying sparrow is a bird → the rule body is unsatisfied → abstain.
        let q = Query::new(t("sparrow", "can", "fly"));
        let a = r.answer(&q);
        assert_eq!(a.provability(), Provability::Abstained);
        assert!(a.value.is_none());
    }

    #[test]
    fn abstains_when_no_rule_covers_the_goal() {
        let r = InductiveReasoner::from_instances(&bird_instances(), &InductionConfig::default());
        let q = q(
            t("sparrow", "can", "teleport"),
            vec![t("sparrow", "is_a", "bird")],
        );
        let a = r.answer(&q);
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn strongest_rule_fires_when_multiple_match() {
        // Two bodies both present for the entity; the higher-confidence rule should be chosen.
        let inst = vec![
            // strong: penguins-in-zoo all fed (3/3)
            t("p1", "in", "zoo"),
            t("p1", "is", "fed"),
            t("p2", "in", "zoo"),
            t("p2", "is", "fed"),
            t("p3", "in", "zoo"),
            t("p3", "is", "fed"),
            // weak: wild ones, only 2/3 fed
            t("p1", "habitat", "wild"),
            t("p4", "habitat", "wild"),
            t("p4", "is", "fed"),
            t("p5", "habitat", "wild"),
            t("p5", "is", "fed"),
            t("p6", "habitat", "wild"),
        ];
        let r = InductiveReasoner::from_instances(&inst, &InductionConfig::default());
        let q = q(
            t("newp", "is", "fed"),
            vec![t("newp", "in", "zoo"), t("newp", "habitat", "wild")],
        );
        let a = r.answer(&q);
        // Whichever fired, it must be heuristic; and the confidence is the stronger (zoo=1.0) rule.
        assert_eq!(a.provability(), Provability::Heuristic);
        if let ProofTrace::Evidence { confidence, .. } = a.proof {
            assert!(
                (confidence - 1.0).abs() < 1e-9,
                "strongest (zoo, conf 1.0) rule should fire"
            );
        } else {
            panic!("expected an Evidence trace");
        }
    }
}
