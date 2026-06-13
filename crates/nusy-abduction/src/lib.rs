//! # nusy-abduction — abductive hypothesis *generation* (EX-4775, VY-E E1)
//!
//! Deduction asks "what follows from the facts?"; **abduction** asks the inverse — "what would have
//! to be true for this observation to follow?". Given an observation `E`, this crate enumerates
//! candidate explanations by **reversing the rules**: for every rule whose *head* unifies with `E`,
//! the (unifier-applied) *body* is a candidate antecedent. Chaining that backward to a bounded depth
//! yields the multi-hop explanations too.
//!
//! Candidates come from the **graph first** ([`GraphCandidates`], [`Substrate::Symbolic`] — a real
//! rule reversal over [`nusy_forward_chain`]), with an **LLM-propose interface behind the same
//! [`CandidateSource`] contract** ([`NeuralProposer`], [`Substrate::Neural`]) that emits *flagged*
//! candidates only — feature-parity stub here, real vLLM wiring is a DGX follow-up. A neural-proposed
//! hypothesis is **never** dressed up as a symbolic derivation: it carries `Substrate::Neural` and
//! its provenance names the proposer, so downstream ranking (E2) and the gate (E3) can treat it as
//! Evidence, never Proof.
//!
//! **Scope (this expedition is generation only).** No scoring, ranking, or consistency-testing of
//! candidates (that is E2/E3); no real LLM calls (mock backend in tests). Arrow-free — FOSS-movable.
//!
//! ```
//! use nusy_abduction::{CandidateSource, GraphCandidates};
//! use nusy_forward_chain::IdRule;
//! use nusy_unify::{Rule, Triple, TriplePattern};
//!
//! // smokes(p) ⊢ at_risk(p, cancer)
//! let rule = IdRule::new(
//!     "risk-from-smoking",
//!     Rule::new(
//!         vec![TriplePattern::parse("?p", "smokes", "true")],
//!         vec![TriplePattern::parse("?p", "at_risk", "cancer")],
//!     ),
//! );
//! let src = GraphCandidates::new(vec![rule], 1);
//! let hs = src.enumerate(&Triple::new("p1", "at_risk", "cancer"));
//! // The abduced explanation: p1 smokes.
//! assert_eq!(hs.len(), 1);
//! assert_eq!(hs[0].ground().unwrap(), vec![Triple::new("p1", "smokes", "true")]);
//! ```

use nusy_forward_chain::IdRule;
use nusy_reasoner::Substrate;
use nusy_unify::{Substitution, Triple, TriplePattern, match_triple};

/// A candidate explanation for an observation: the antecedent atoms that, if true, would make the
/// observation follow — together with how the candidate was generated.
///
/// Atoms are [`TriplePattern`]s: a fully-ground atom is a concrete proposed fact, while a remaining
/// free variable is an **existential** ("*some* `?c` such that …") — the honest shape of an abductive
/// guess. Use [`Hypothesis::ground`] to get concrete triples when every atom is bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hypothesis {
    /// The explanatory antecedent — what would have to hold for the observation to follow.
    pub explanation: Vec<TriplePattern>,
    /// Generation provenance: the rule-id chain (symbolic, oldest→newest) or the proposer id (neural).
    pub provenance: Vec<String>,
    /// Where this candidate came from — [`Substrate::Symbolic`] (rule reversal) or
    /// [`Substrate::Neural`] (a flagged proposer). Never authoritative on its own; abduction
    /// *generates*, it does not prove.
    pub substrate: Substrate,
}

impl Hypothesis {
    /// The explanation as ground triples, or `None` if any atom still carries a free variable
    /// (an unbound existential).
    pub fn ground(&self) -> Option<Vec<Triple>> {
        let id = Substitution::new();
        self.explanation.iter().map(|p| id.ground(p)).collect()
    }
}

/// A source of abductive candidates for an observation. The graph enumerator and the neural proposer
/// implement the same contract so a [`crate::Pipeline`](nusy_reasoner)-style consumer can fan over
/// both and keep each candidate's substrate tag.
pub trait CandidateSource {
    /// Enumerate candidate explanations for `observation`.
    fn enumerate(&self, observation: &Triple) -> Vec<Hypothesis>;
    /// The substrate of the candidates this source emits.
    fn substrate(&self) -> Substrate;
}

/// **Symbolic abduction by rule reversal.** For each rule whose head unifies with the observation,
/// the head-unifier-applied body is a candidate explanation; chaining backward to `max_depth`
/// enumerates multi-hop explanations. Pure graph enumeration — no LLM, no ranking.
#[derive(Debug, Clone)]
pub struct GraphCandidates {
    rules: Vec<IdRule>,
    max_depth: usize,
}

impl GraphCandidates {
    /// Build over a rule set, abducing backward up to `max_depth` hops (`max_depth` 1 = direct
    /// rule reversal only; higher chains through reducible sub-goals). `max_depth` 0 yields nothing.
    pub fn new(rules: Vec<IdRule>, max_depth: usize) -> Self {
        Self { rules, max_depth }
    }

    /// All backward reductions of `goal` within `depth` hops. Each result is (explanation atoms,
    /// rule-id chain). Includes the trivial un-reduced `[goal]` with an empty chain (the caller
    /// drops that at the top level — an explanation must reverse at least one rule).
    fn reduce(&self, goal: &TriplePattern, depth: usize) -> Vec<(Vec<TriplePattern>, Vec<String>)> {
        // The goal can always stand un-reduced (it becomes a leaf of the explanation).
        let mut out: Vec<(Vec<TriplePattern>, Vec<String>)> =
            vec![(vec![goal.clone()], Vec::new())];
        if depth == 0 {
            return out;
        }
        let Some(goal_fact) = Substitution::new().ground(goal) else {
            // Only ground sub-goals are reduced further; an existential leaf stays a leaf.
            return out;
        };
        for r in &self.rules {
            for head in &r.rule.rhs {
                let Some(s) = match_triple(head, &goal_fact, &Substitution::new()) else {
                    continue;
                };
                // The body, instantiated by the head unifier, is this rule's reduction of the goal.
                let body: Vec<TriplePattern> =
                    r.rule.lhs.iter().map(|p| s.apply_pattern(p)).collect();
                // Cartesian-combine each body atom's own reductions (depth-1) so multi-hop chains
                // are enumerated; an atom with no reduction contributes only itself.
                for (atoms, mut chain) in self.combine_atoms(&body, depth - 1) {
                    chain.push(r.id.clone());
                    out.push((atoms, chain));
                }
            }
        }
        out
    }

    /// Cartesian product of the per-atom reductions of a conjunction, merging leaf-lists and
    /// concatenating rule chains (oldest sub-goal first).
    fn combine_atoms(
        &self,
        atoms: &[TriplePattern],
        depth: usize,
    ) -> Vec<(Vec<TriplePattern>, Vec<String>)> {
        let mut acc: Vec<(Vec<TriplePattern>, Vec<String>)> = vec![(Vec::new(), Vec::new())];
        for atom in atoms {
            let reductions = self.reduce(atom, depth);
            let mut next = Vec::new();
            for (leaves, chain) in &acc {
                for (r_leaves, r_chain) in &reductions {
                    let mut l = leaves.clone();
                    l.extend(r_leaves.iter().cloned());
                    let mut c = chain.clone();
                    c.extend(r_chain.iter().cloned());
                    next.push((l, c));
                }
            }
            acc = next;
        }
        acc
    }
}

impl CandidateSource for GraphCandidates {
    fn enumerate(&self, observation: &Triple) -> Vec<Hypothesis> {
        let goal = TriplePattern::new(
            nusy_unify::Term::Const(observation.subject.clone()),
            nusy_unify::Term::Const(observation.predicate.clone()),
            nusy_unify::Term::Const(observation.object.clone()),
        );
        let mut seen = Vec::new();
        let mut out = Vec::new();
        for (atoms, chain) in self.reduce(&goal, self.max_depth) {
            // Drop the trivial self-explanation (no rule reversed) and exact duplicates.
            if chain.is_empty() {
                continue;
            }
            let key = (atoms.clone(), chain.clone());
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            out.push(Hypothesis {
                explanation: atoms,
                provenance: chain,
                substrate: Substrate::Symbolic,
            });
        }
        out
    }

    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }
}

/// The interface a real LLM backend implements to propose candidate explanations. Kept minimal and
/// behind a trait so the real vLLM wiring (DGX follow-up) drops in without touching the contract.
pub trait Propose {
    /// Propose zero or more candidate antecedents (each a conjunction of ground triples) for the
    /// observation. Best-effort — these are *guesses*, surfaced as flagged neural candidates.
    fn propose(&self, observation: &Triple) -> Vec<Vec<Triple>>;
    /// An id for this proposer, recorded in each candidate's provenance.
    fn proposer_id(&self) -> &str;
}

/// **Neural abduction proposer** behind the [`CandidateSource`] contract. Wraps a [`Propose`]
/// backend and tags every candidate [`Substrate::Neural`] with `neural:<proposer>` provenance — so
/// nothing it emits can be mistaken for a symbolic derivation. Real model calls are a follow-up;
/// tests use a mock backend.
#[derive(Debug, Clone)]
pub struct NeuralProposer<B: Propose> {
    backend: B,
}

impl<B: Propose> NeuralProposer<B> {
    /// Wrap a propose backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: Propose> CandidateSource for NeuralProposer<B> {
    fn enumerate(&self, observation: &Triple) -> Vec<Hypothesis> {
        let tag = format!("neural:{}", self.backend.proposer_id());
        self.backend
            .propose(observation)
            .into_iter()
            .map(|triples| Hypothesis {
                explanation: triples.into_iter().map(triple_to_pattern).collect(),
                provenance: vec![tag.clone()],
                substrate: Substrate::Neural,
            })
            .collect()
    }

    fn substrate(&self) -> Substrate {
        Substrate::Neural
    }
}

/// A ground triple as an all-`Const` pattern.
fn triple_to_pattern(t: Triple) -> TriplePattern {
    TriplePattern::new(
        nusy_unify::Term::Const(t.subject),
        nusy_unify::Term::Const(t.predicate),
        nusy_unify::Term::Const(t.object),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_unify::Rule;

    fn idrule(id: &str, body: Vec<TriplePattern>, head: Vec<TriplePattern>) -> IdRule {
        IdRule::new(id, Rule::new(body, head))
    }

    /// Direct rule reversal: observation unifies with a head → the body is the explanation.
    #[test]
    fn direct_reversal_enumerates_the_body_as_explanation() {
        let r = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let src = GraphCandidates::new(vec![r], 1);
        let hs = src.enumerate(&Triple::new("p1", "at_risk", "cancer"));

        assert_eq!(hs.len(), 1);
        assert_eq!(hs[0].substrate, Substrate::Symbolic);
        assert_eq!(hs[0].provenance, vec!["risk-from-smoking"]);
        assert_eq!(
            hs[0].ground().unwrap(),
            vec![Triple::new("p1", "smokes", "true")]
        );
    }

    /// Multi-hop: a 2-rule chain (condition→frail→at-risk) must enumerate BOTH the one-hop
    /// explanation (frail) and the two-hop one (the condition), each with its full rule chain.
    /// This is the enumeration-completeness property: every gold explanation appears.
    #[test]
    fn bounded_depth_enumerates_every_explanation_in_a_chain() {
        let frail = idrule(
            "frail-from-condition",
            vec![TriplePattern::parse("?p", "has_condition", "osteoporosis")],
            vec![TriplePattern::parse("?p", "frail", "true")],
        );
        let at_risk = idrule(
            "at-risk-from-frail",
            vec![TriplePattern::parse("?p", "frail", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "fall")],
        );
        let src = GraphCandidates::new(vec![frail, at_risk], 3);
        let hs = src.enumerate(&Triple::new("p1", "at_risk", "fall"));

        let grounds: Vec<(Vec<Triple>, Vec<String>)> = hs
            .iter()
            .filter_map(|h| h.ground().map(|g| (g, h.provenance.clone())))
            .collect();

        // One-hop: p1 frail, via the at-risk rule.
        assert!(
            grounds
                .iter()
                .any(|(g, c)| g == &vec![Triple::new("p1", "frail", "true")]
                    && c == &vec!["at-risk-from-frail".to_string()]),
            "missing one-hop explanation: {grounds:?}"
        );
        // Two-hop: p1 has osteoporosis, via the full chain (frail rule then at-risk rule).
        assert!(
            grounds.iter().any(|(g, c)| g
                == &vec![Triple::new("p1", "has_condition", "osteoporosis")]
                && c == &vec![
                    "frail-from-condition".to_string(),
                    "at-risk-from-frail".to_string()
                ]),
            "missing two-hop chained explanation: {grounds:?}"
        );
    }

    /// An observation no rule head unifies with → no symbolic candidates (abstain, never invent).
    #[test]
    fn unexplainable_observation_yields_no_candidates() {
        let r = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let src = GraphCandidates::new(vec![r], 2);
        assert!(src.enumerate(&Triple::new("p1", "likes", "tea")).is_empty());
    }

    /// An existential explanation: the rule body carries a variable the head doesn't bind, so the
    /// abduced atom stays a free pattern (not a ground triple) — surfaced honestly, not dropped.
    #[test]
    fn unbound_body_variable_is_an_existential_pattern() {
        let r = idrule(
            "frail-from-some-condition",
            vec![
                TriplePattern::parse("?p", "has_condition", "?c"),
                TriplePattern::parse("?c", "increases_fall_risk", "true"),
            ],
            vec![TriplePattern::parse("?p", "frail", "true")],
        );
        let src = GraphCandidates::new(vec![r], 1);
        let hs = src.enumerate(&Triple::new("p1", "frail", "true"));

        assert_eq!(hs.len(), 1);
        // ?c is unbound → not groundable, but the patterns are present (p1 has_condition ?c, etc.).
        assert!(hs[0].ground().is_none());
        assert_eq!(hs[0].explanation.len(), 2);
        assert_eq!(
            hs[0].explanation[0],
            TriplePattern::parse("p1", "has_condition", "?c")
        );
    }

    struct MockProposer;
    impl Propose for MockProposer {
        fn propose(&self, observation: &Triple) -> Vec<Vec<Triple>> {
            // A canned "guess": whatever the subject is, propose it has an unknown viral cause.
            vec![vec![Triple::new(
                &observation.subject,
                "has_cause",
                "virus",
            )]]
        }
        fn proposer_id(&self) -> &str {
            "mock-llm"
        }
    }

    /// The neural proposer emits flagged, Neural-substrate candidates — never a symbolic
    /// derivation. Provenance names the proposer, so E2/E3 treat it as Evidence, not Proof.
    #[test]
    fn neural_proposer_emits_only_flagged_neural_candidates() {
        let src = NeuralProposer::new(MockProposer);
        assert_eq!(src.substrate(), Substrate::Neural);

        let hs = src.enumerate(&Triple::new("p1", "at_risk", "fall"));
        assert_eq!(hs.len(), 1);
        let h = &hs[0];
        assert_eq!(h.substrate, Substrate::Neural, "must be flagged Neural");
        assert_eq!(h.provenance, vec!["neural:mock-llm"]);
        assert_eq!(
            h.ground().unwrap(),
            vec![Triple::new("p1", "has_cause", "virus")]
        );
        // No symbolic provenance (no rule id) ever appears on a neural candidate.
        assert!(
            !h.provenance.iter().any(|p| !p.starts_with("neural:")),
            "a neural candidate must carry only neural provenance, never a rule id"
        );
    }

    /// max_depth 0 reduces nothing — a guard that the depth bound is honoured.
    #[test]
    fn zero_depth_enumerates_nothing() {
        let r = idrule(
            "risk-from-smoking",
            vec![TriplePattern::parse("?p", "smokes", "true")],
            vec![TriplePattern::parse("?p", "at_risk", "cancer")],
        );
        let src = GraphCandidates::new(vec![r], 0);
        assert!(
            src.enumerate(&Triple::new("p1", "at_risk", "cancer"))
                .is_empty()
        );
    }
}
