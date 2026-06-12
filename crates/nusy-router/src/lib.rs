//! # nusy-router — route/no-route + rule-path classifier (EX-4611, VOY-V18-5)
//!
//! The pre- and post-filter wrapped around the [`nusy_gate::ProvableClaimGate`]:
//!
//! - **Pre-filter** — [`RouteClassifier`] decides whether a claim is *worth* invoking the
//!   engine on. A claim whose predicate the engine knows about (it appears in the saturated
//!   fact set, or in any rule body/head) is routed to the gate ([`RouteDecision::Symbolic`]);
//!   a claim whose predicate is foreign is routed straight to the neural layer
//!   ([`RouteDecision::Neural`]) without spending a proof attempt.
//! - **Post-filter** — [`RulePath`] surfaces the **rule-id path** of a Proven response, the
//!   stable identifier of the rule chain that justified it. Lets callers classify, tally, or
//!   dedup proofs by which rules were used (the original V17 "rule/path-id" abstraction, now
//!   riding the proof API rather than a learned hidden-state classifier — §11.2).
//!
//! ## Why pre-filter?
//!
//! After EX-4610 landed, "is a claim provable?" *is* "does the engine return a proof for it?"
//! — the proof-attempt is the router (Mini, EX-4611 design note, 2026-06-10). But invoking the
//! engine on a claim whose predicate isn't even in our schema is wasted work. The pre-filter
//! is the cheap "do we have anything to say about this predicate?" check that lets the caller
//! skip the engine on out-of-domain claims (e.g., a clinical engine asked about the weather).
//!
//! For in-domain claims the pre-filter says [`RouteDecision::Symbolic`] and the caller hands
//! the claim to the gate; the gate's verdict — Proven or Unproven — is then authoritative.
//!
//! ## Example
//!
//! ```
//! use nusy_forward_chain::{IdRule, forward_chain};
//! use nusy_gate::ProvableClaimGate;
//! use nusy_router::{RouteClassifier, RouteDecision, RulePath};
//! use nusy_unify::{Rule, Triple, TriplePattern};
//!
//! let rule = IdRule::new("at-risk-fall", Rule::new(
//!     vec![TriplePattern::parse("?p", "has_condition", "?c"),
//!          TriplePattern::parse("?c", "increases_fall_risk", "true")],
//!     vec![TriplePattern::parse("?p", "at_risk", "fall")]));
//! let rules = vec![rule];
//! let facts = vec![
//!     Triple::new("p1", "has_condition", "osteoporosis"),
//!     Triple::new("osteoporosis", "increases_fall_risk", "true"),
//! ];
//! let sat = forward_chain(&rules, facts);
//! let gate = ProvableClaimGate::new(sat.clone());
//! let router = RouteClassifier::from_engine(&sat, &rules);
//!
//! // A claim whose predicate the engine knows about → invoke the gate.
//! let claim = Triple::new("p1", "at_risk", "fall");
//! assert!(matches!(router.classify(&claim), RouteDecision::Symbolic));
//! let resp = gate.gate(&claim);
//! assert!(resp.is_proven());
//! // Surface the rule path of the proof.
//! let path = RulePath::from_response(&resp).expect("proven → has a path");
//! assert_eq!(path.rule_ids, vec!["at-risk-fall"]);
//!
//! // A claim whose predicate the engine has no rules or facts for → skip the gate.
//! let foreign = Triple::new("p1", "weather_today", "sunny");
//! assert!(matches!(router.classify(&foreign), RouteDecision::Neural { .. }));
//! ```

pub mod reasoner_router;
pub use reasoner_router::{AnswerClass, ParReport, ReasonerRouter, RouteOutcome, RoutedVerdict};

use std::collections::HashSet;

use nusy_forward_chain::{IdRule, Saturation};
use nusy_gate::GateResponse;
use nusy_unify::{Term, Triple};

/// Where to route a candidate claim — to the symbolic gate, or directly to the neural layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    /// The claim's predicate is in the engine's symbolic domain — invoke the gate next.
    Symbolic,
    /// The claim's predicate is unknown to the engine — skip the gate; route to neural.
    Neural {
        /// Why the pre-filter routed away from the gate (human-readable, for logging/eval).
        reason: String,
    },
}

impl RouteDecision {
    /// Was the claim routed to the symbolic gate?
    pub fn is_symbolic(&self) -> bool {
        matches!(self, RouteDecision::Symbolic)
    }

    /// Was the claim routed to the neural layer?
    pub fn is_neural(&self) -> bool {
        matches!(self, RouteDecision::Neural { .. })
    }
}

/// Tally of a batch of routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RouteSummary {
    /// Claims routed to the symbolic gate.
    pub symbolic: usize,
    /// Claims routed straight to neural (predicate not in engine's domain).
    pub neural: usize,
}

/// The route/no-route pre-filter for the provable-claim gate.
///
/// Holds the set of predicates the engine has anything to say about (any predicate
/// appearing in a saturated fact, or — when built from a rule set — any constant
/// predicate in a rule body or head). A claim whose predicate is in this set is routed
/// [`RouteDecision::Symbolic`]; everything else is routed [`RouteDecision::Neural`].
///
/// If the rule schema contains a meta-rule with a *variable* predicate (the rule
/// matches any predicate), the classifier records `accept_all = true` — pre-filtering
/// can no longer rule any predicate out, so every claim is [`RouteDecision::Symbolic`].
#[derive(Debug, Clone)]
pub struct RouteClassifier {
    known_predicates: HashSet<String>,
    accept_all: bool,
}

impl RouteClassifier {
    /// Build the classifier from the engine's saturation: the predicate of every fact
    /// (seed or derived) is in the engine's domain.
    pub fn from_saturation(sat: &Saturation) -> Self {
        let known_predicates: HashSet<String> =
            sat.facts.iter().map(|t| t.predicate.clone()).collect();
        Self {
            known_predicates,
            accept_all: false,
        }
    }

    /// Build the classifier from a rule set: every constant predicate appearing in a
    /// rule body or head is in the engine's domain. A variable predicate in any rule
    /// flips the classifier to accept-all (a meta-rule that matches any predicate).
    pub fn from_rules(rules: &[IdRule]) -> Self {
        let mut known_predicates: HashSet<String> = HashSet::new();
        let mut accept_all = false;
        for r in rules {
            for pat in r.rule.lhs.iter().chain(r.rule.rhs.iter()) {
                match &pat.predicate {
                    Term::Const(c) => {
                        known_predicates.insert(c.clone());
                    }
                    Term::Var(_) => {
                        accept_all = true;
                    }
                }
            }
        }
        Self {
            known_predicates,
            accept_all,
        }
    }

    /// Build the classifier from BOTH saturated facts AND the rule schema — the union
    /// of [`from_saturation`](Self::from_saturation) and [`from_rules`](Self::from_rules).
    /// This is the recommended constructor: it covers predicates the engine has facts
    /// about *today* and predicates it has rules for (could derive facts about tomorrow).
    pub fn from_engine(sat: &Saturation, rules: &[IdRule]) -> Self {
        let mut s = Self::from_saturation(sat);
        let r = Self::from_rules(rules);
        s.known_predicates.extend(r.known_predicates);
        s.accept_all |= r.accept_all;
        s
    }

    /// Build the classifier from an explicit set of predicates — useful when the
    /// caller already knows the schema (e.g. from Y2 ontology) and wants to gate
    /// without first running the engine.
    pub fn with_predicates<I, S>(preds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            known_predicates: preds.into_iter().map(Into::into).collect(),
            accept_all: false,
        }
    }

    /// Does the classifier accept any predicate (i.e. is the pre-filter effectively
    /// disabled, because a meta-rule matches any predicate)?
    pub fn is_accept_all(&self) -> bool {
        self.accept_all
    }

    /// The predicates the engine has anything to say about.
    pub fn known_predicates(&self) -> impl Iterator<Item = &str> {
        self.known_predicates.iter().map(String::as_str)
    }

    /// How many predicates are in the engine's domain.
    pub fn known_predicate_count(&self) -> usize {
        self.known_predicates.len()
    }

    /// Route one claim. `Symbolic` if its predicate is in the engine's domain (or the
    /// classifier is accept-all); `Neural { reason }` otherwise.
    pub fn classify(&self, claim: &Triple) -> RouteDecision {
        if self.accept_all || self.known_predicates.contains(&claim.predicate) {
            RouteDecision::Symbolic
        } else {
            RouteDecision::Neural {
                reason: format!(
                    "predicate `{}` is not in the engine's symbolic domain",
                    claim.predicate
                ),
            }
        }
    }

    /// Route a batch of claims, preserving order.
    pub fn classify_all(&self, claims: &[Triple]) -> Vec<RouteDecision> {
        claims.iter().map(|c| self.classify(c)).collect()
    }

    /// Tally a batch: how many would the gate be invoked on vs. routed straight to neural.
    pub fn summarize(&self, claims: &[Triple]) -> RouteSummary {
        let mut s = RouteSummary::default();
        for c in claims {
            if self.classify(c).is_symbolic() {
                s.symbolic += 1;
            } else {
                s.neural += 1;
            }
        }
        s
    }
}

/// The rule-id path of a Proven response — a stable identifier of which rules were
/// chained together to derive the claim.
///
/// The list is in pre-order: the outermost rule (the one whose head matched the claim)
/// first, then the rules used to prove its premises, depth-first. An axiom-only proof
/// (a seed fact) has an empty path. Two proofs with the same `rule_ids` walked the same
/// rule chain — the V17 "rule/path-id" classification, now surfaced from the proof API.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct RulePath {
    /// Rule IDs in pre-order traversal of the proof tree (root rule first).
    pub rule_ids: Vec<String>,
}

impl RulePath {
    /// Extract the rule-id path from a Proven response. `None` for an Unproven response.
    pub fn from_response(resp: &GateResponse) -> Option<Self> {
        resp.proof().map(Self::from_proof)
    }

    /// Extract the rule-id path directly from a proof tree.
    pub fn from_proof(proof: &nusy_forward_chain::ProofTree) -> Self {
        Self {
            rule_ids: proof.rule_ids().into_iter().map(str::to_string).collect(),
        }
    }

    /// Build a path from an explicit rule-id list (useful in tests and downstream callers).
    pub fn new<I, S>(rule_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            rule_ids: rule_ids.into_iter().map(Into::into).collect(),
        }
    }

    /// Number of rule applications in the path (0 = axiom-only proof).
    pub fn depth(&self) -> usize {
        self.rule_ids.len()
    }

    /// Is this an axiom-only proof (no rule applications)?
    pub fn is_axiom_only(&self) -> bool {
        self.rule_ids.is_empty()
    }

    /// A canonical string form — rule IDs joined by `>` in pre-order. Suitable as a
    /// metrics-key or dedup-key (`"recommend-fall-assessment>at-risk-fall"`).
    pub fn id(&self) -> String {
        self.rule_ids.join(">")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_forward_chain::{IdRule, forward_chain};
    use nusy_unify::{Rule, TriplePattern};

    fn t(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn at_risk_rule() -> IdRule {
        IdRule::new(
            "at-risk-fall",
            Rule::new(
                vec![
                    TriplePattern::parse("?p", "has_condition", "?c"),
                    TriplePattern::parse("?c", "increases_fall_risk", "true"),
                ],
                vec![TriplePattern::parse("?p", "at_risk", "fall")],
            ),
        )
    }

    #[test]
    fn route_decision_basic_predicates() {
        let symbolic = RouteDecision::Symbolic;
        assert!(symbolic.is_symbolic());
        assert!(!symbolic.is_neural());

        let neural = RouteDecision::Neural { reason: "x".into() };
        assert!(neural.is_neural());
        assert!(!neural.is_symbolic());
    }

    #[test]
    fn from_saturation_collects_seed_and_derived_predicates() {
        let facts = vec![
            t("p1", "has_condition", "osteoporosis"),
            t("osteoporosis", "increases_fall_risk", "true"),
        ];
        let sat = forward_chain(&[at_risk_rule()], facts);
        let cls = RouteClassifier::from_saturation(&sat);
        // Seed predicates.
        assert!(cls.known_predicates().any(|p| p == "has_condition"));
        assert!(cls.known_predicates().any(|p| p == "increases_fall_risk"));
        // Derived predicate (the rule fired).
        assert!(cls.known_predicates().any(|p| p == "at_risk"));
        assert!(!cls.is_accept_all());
    }

    #[test]
    fn from_rules_collects_body_and_head_predicates() {
        let cls = RouteClassifier::from_rules(&[at_risk_rule()]);
        let preds: HashSet<&str> = cls.known_predicates().collect();
        assert!(preds.contains("has_condition"));
        assert!(preds.contains("increases_fall_risk"));
        assert!(preds.contains("at_risk"));
        assert!(!cls.is_accept_all());
    }

    #[test]
    fn from_engine_unions_facts_and_rules() {
        // A rule that hasn't fired (no matching seed facts) — its head predicate is
        // only visible through the rule schema, not through the saturation.
        let dormant = IdRule::new(
            "dormant",
            Rule::new(
                vec![TriplePattern::parse("?x", "never_holds", "?y")],
                vec![TriplePattern::parse("?x", "dormant_head", "?y")],
            ),
        );
        let sat = forward_chain(
            std::slice::from_ref(&dormant),
            vec![t("a", "seed_pred", "b")],
        );
        let from_sat = RouteClassifier::from_saturation(&sat);
        // Saturation alone sees only the seed predicate.
        assert!(from_sat.known_predicates().any(|p| p == "seed_pred"));
        assert!(!from_sat.known_predicates().any(|p| p == "dormant_head"));

        let from_eng = RouteClassifier::from_engine(&sat, &[dormant]);
        // Engine view sees seed AND dormant rule's head/body.
        let preds: HashSet<&str> = from_eng.known_predicates().collect();
        assert!(preds.contains("seed_pred"));
        assert!(preds.contains("dormant_head"));
        assert!(preds.contains("never_holds"));
    }

    #[test]
    fn variable_predicate_rule_flips_accept_all() {
        // A meta-rule with a variable predicate position: matches any predicate.
        let meta = IdRule::new(
            "meta",
            Rule::new(
                vec![TriplePattern::parse("?x", "?p", "?y")],
                vec![TriplePattern::parse("?x", "echoed", "?y")],
            ),
        );
        let cls = RouteClassifier::from_rules(&[meta]);
        assert!(cls.is_accept_all());
        // Any predicate is routed Symbolic — pre-filter can't rule any out.
        assert!(
            cls.classify(&t("any", "completely_unknown", "anything"))
                .is_symbolic()
        );
    }

    #[test]
    fn classify_routes_known_and_unknown_predicates() {
        let cls = RouteClassifier::with_predicates(["parent", "ancestor"]);
        assert!(cls.classify(&t("a", "parent", "b")).is_symbolic());
        match cls.classify(&t("a", "weather_today", "sunny")) {
            RouteDecision::Neural { reason } => {
                assert!(reason.contains("weather_today"));
                assert!(reason.contains("symbolic domain"));
            }
            RouteDecision::Symbolic => panic!("unknown predicate must be Neural"),
        }
    }

    #[test]
    fn batch_classify_preserves_order_and_summarizes() {
        let cls = RouteClassifier::with_predicates(["parent"]);
        let claims = vec![
            t("a", "parent", "b"),      // symbolic
            t("a", "weather", "sunny"), // neural
            t("c", "parent", "d"),      // symbolic
        ];
        let decisions = cls.classify_all(&claims);
        assert_eq!(decisions.len(), 3);
        assert!(decisions[0].is_symbolic());
        assert!(decisions[1].is_neural());
        assert!(decisions[2].is_symbolic());

        let summary = cls.summarize(&claims);
        assert_eq!(summary.symbolic, 2);
        assert_eq!(summary.neural, 1);
    }

    #[test]
    fn rule_path_from_proof_is_preorder() {
        // grandparent ← parent + parent — two-step proof, root rule first.
        let parent_facts = vec![t("a", "parent", "b"), t("b", "parent", "c")];
        let grandparent = IdRule::new(
            "grandparent",
            Rule::new(
                vec![
                    TriplePattern::parse("?x", "parent", "?y"),
                    TriplePattern::parse("?y", "parent", "?z"),
                ],
                vec![TriplePattern::parse("?x", "grandparent", "?z")],
            ),
        );
        let sat = forward_chain(&[grandparent], parent_facts);
        let proof = sat
            .proof_of(&t("a", "grandparent", "c"))
            .expect("grandparent derivable");
        let path = RulePath::from_proof(&proof);
        assert_eq!(path.rule_ids, vec!["grandparent"]);
        assert_eq!(path.depth(), 1);
        assert!(!path.is_axiom_only());
        assert_eq!(path.id(), "grandparent");
    }

    #[test]
    fn rule_path_from_response_round_trips_proven_and_unproven() {
        let facts = vec![t("a", "parent", "b")];
        let sat = forward_chain(&[], facts);
        let gate = nusy_gate::ProvableClaimGate::new(sat);

        // A seed fact is provable as an axiom — empty rule path.
        let resp = gate.gate(&t("a", "parent", "b"));
        let path = RulePath::from_response(&resp).expect("proven");
        assert!(path.is_axiom_only());
        assert_eq!(path.depth(), 0);
        assert_eq!(path.id(), "");

        // An unprovable claim has no path.
        let resp = gate.gate(&t("a", "parent", "z"));
        assert!(RulePath::from_response(&resp).is_none());
    }

    #[test]
    fn rule_path_equality_and_hash_are_by_id_sequence() {
        let a = RulePath::new(["r1", "r2"]);
        let b = RulePath::new(["r1", "r2"]);
        let c = RulePath::new(["r2", "r1"]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Hashable — can be a HashMap key for rule-path tallying.
        let mut counts: std::collections::HashMap<RulePath, usize> =
            std::collections::HashMap::new();
        *counts.entry(a.clone()).or_insert(0) += 1;
        *counts.entry(b).or_insert(0) += 1;
        *counts.entry(c).or_insert(0) += 1;
        assert_eq!(counts[&a], 2);
        assert_eq!(counts.len(), 2);
    }
}
