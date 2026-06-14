//! # nusy-case-based — the case-based reasoner (VY-B E3, EX-4889)
//!
//! Reasoning by **retrieve-and-adapt**: keep a library of prior *cases*, each pairing a **problem**
//! (the features that described a past situation) with a **solution** (what was done / what held).
//! Given a new target, **retrieve** the most similar prior case by feature overlap, then **adapt**
//! its solution onto the target. This is the engine behind incident-pattern playbooks and clinical
//! case libraries — "this looks like the case we saw before, so try what worked then."
//!
//! Case-based reasoning is *retrieval over a feature vector*, distinct from the relational
//! structure-mapping of [`nusy-analogical`](../nusy_analogical): CBR asks "which past case most
//! resembles this one?" and reuses that case's recorded solution, rather than aligning relational
//! structure between two specific situations.
//!
//! ## The guarantee invariant — a retrieved precedent is HEURISTIC, never Proven
//!
//! "A past case resembles yours" is evidence, not proof: the precedent might differ in the one
//! feature that matters. So every answer this reasoner produces carries a [`ProofTrace::Evidence`]
//! trace whose [`provability`](nusy_reasoner::Answer::provability) is *always*
//! [`Provability::Heuristic`](nusy_reasoner::Provability::Heuristic). The engine holds no
//! [`DerivationTrace`](nusy_reasoner::DerivationTrace), so it is **structurally unable to mint
//! `Proven`** — exactly as the Reasoner contract requires. Retrieved-and-adapted solutions feed
//! downstream reasoning as *candidates*; only a sound deductive engine can certify them `Proven`.
//!
//! ## Generic by construction
//!
//! The engine contains **no domain literals** — every feature and solution it retrieves and adapts
//! comes from the case library and the query, never from the code. The same engine serves a clinical
//! case library, an incident-pattern playbook, or a legal case file (the tests prove it).
//!
//! ```
//! use nusy_case_based::{Case, CaseBasedReasoner, CbrConfig};
//! use nusy_reasoner::{Provability, Query, Reasoner};
//! use nusy_unify::Triple;
//!
//! // A prior case: a patient with fever + cough was tested for flu.
//! let prior = Case::new(
//!     "case-1",
//!     vec![("has_symptom", "fever"), ("has_symptom", "cough")], // problem features
//!     vec![("order", "flu_test")],                              // recorded solution
//! );
//! let r = CaseBasedReasoner::new(vec![prior], CbrConfig::default());
//!
//! // A new patient with the same features: retrieve the case, adapt its solution — heuristically.
//! let q = Query {
//!     goal: Triple::new("patient_b", "order", "flu_test"),
//!     context: vec![
//!         Triple::new("patient_b", "has_symptom", "fever"),
//!         Triple::new("patient_b", "has_symptom", "cough"),
//!     ],
//! };
//! let a = r.answer(&q);
//! assert_eq!(a.value, Some(Triple::new("patient_b", "order", "flu_test")));
//! assert_eq!(a.provability(), Provability::Heuristic); // a precedent is not a proof
//! assert_ne!(a.provability(), Provability::Proven);
//! ```

use std::collections::BTreeSet;

use nusy_reasoner::{
    Answer, CompetenceEnvelope, Guarantee, ProofTrace, Query, QueryShape, Reasoner, Substrate,
};
use nusy_unify::Triple;

/// A `(predicate, object)` feature — the unit a case's problem and solution are described in.
pub type Feature = (String, String);

/// A prior case: a **problem** (the features that characterized a past situation) paired with a
/// **solution** (the `(predicate, object)` outcomes that were applied / held).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    /// Stable identifier, used in provenance (e.g. `"case-1"`).
    pub name: String,
    /// The problem features — what made this case what it is. Retrieval scores against these.
    pub problem: BTreeSet<Feature>,
    /// The recorded solution — `(predicate, object)` outcomes adaptation transfers to the target.
    pub solution: BTreeSet<Feature>,
}

impl Case {
    /// Construct a case from a name, its problem features, and its recorded solution.
    pub fn new(
        name: impl Into<String>,
        problem: Vec<(&str, &str)>,
        solution: Vec<(&str, &str)>,
    ) -> Self {
        Self {
            name: name.into(),
            problem: problem
                .into_iter()
                .map(|(p, o)| (p.to_string(), o.to_string()))
                .collect(),
            solution: solution
                .into_iter()
                .map(|(p, o)| (p.to_string(), o.to_string()))
                .collect(),
        }
    }
}

/// Thresholds governing retrieval.
#[derive(Debug, Clone, Copy)]
pub struct CbrConfig {
    /// Minimum Jaccard similarity between target features and a case's problem for retrieval to
    /// fire ∈ (0, 1]. Guards against adapting a case that barely resembles the target.
    pub min_similarity: f64,
}

impl Default for CbrConfig {
    /// Conservative default: the target must share a majority (Jaccard > 0.5) of features with the
    /// retrieved case's problem.
    fn default() -> Self {
        Self { min_similarity: 0.5 }
    }
}

/// Jaccard similarity of two feature sets: `|A ∩ B| / |A ∪ B|` ∈ [0, 1]. Symmetric; 1.0 iff equal,
/// 0.0 iff disjoint. Two empty sets are treated as dissimilar (0.0) — an empty problem retrieves
/// nothing.
pub fn jaccard(a: &BTreeSet<Feature>, b: &BTreeSet<Feature>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    inter as f64 / union as f64
}

/// A [`Reasoner`] that answers by **retrieving** the most similar prior case and **adapting** its
/// solution. Every answer is [`Provability::Heuristic`](nusy_reasoner::Provability) — a precedent
/// resembling the target is evidence, not proof.
pub struct CaseBasedReasoner {
    cases: Vec<Case>,
    cfg: CbrConfig,
    envelope: CompetenceEnvelope,
}

impl CaseBasedReasoner {
    /// Build a reasoner over a case library. Its competence covers exactly the predicates the cases'
    /// solutions could adapt (the predicates appearing in their solutions).
    pub fn new(cases: Vec<Case>, cfg: CbrConfig) -> Self {
        let predicates: BTreeSet<String> = cases
            .iter()
            .flat_map(|c| c.solution.iter().map(|(p, _)| p.clone()))
            .collect();
        let envelope = CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "retrieved-and-adapted case".into(),
                predicates: predicates.into_iter().collect(),
            }],
        };
        Self {
            cases,
            cfg,
            envelope,
        }
    }

    /// The case library this reasoner draws on.
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// The target's feature set: the `(predicate, object)` facts in `context` about the goal's
    /// subject. These are scored against each case's problem.
    fn target_features(goal: &Triple, context: &[Triple]) -> BTreeSet<Feature> {
        context
            .iter()
            .filter(|t| t.subject == goal.subject)
            .map(|t| (t.predicate.clone(), t.object.clone()))
            .collect()
    }
}

impl Reasoner for CaseBasedReasoner {
    /// Retrieve the most similar prior case (by Jaccard over features), and if it clears
    /// [`CbrConfig::min_similarity`] AND its solution contains the goal relation, adapt that solution
    /// onto the target as **evidence** (never a derivation). Abstains when no case is similar enough
    /// or the nearest case's solution does not cover the goal. Ties broken by case name, for
    /// determinism.
    fn answer(&self, query: &Query) -> Answer {
        let target = Self::target_features(&query.goal, &query.context);
        if target.is_empty() {
            return Answer::abstained(); // no features to match on
        }

        // Retrieve: the most similar case by Jaccard, tie-broken by name.
        let mut best: Option<(f64, &Case)> = None;
        for case in &self.cases {
            let sim = jaccard(&target, &case.problem);
            let better = match &best {
                None => true,
                Some((bsim, bcase)) => sim > *bsim || (sim == *bsim && case.name < bcase.name),
            };
            if better {
                best = Some((sim, case));
            }
        }

        let (sim, case) = match best {
            Some(b) => b,
            None => return Answer::abstained(),
        };
        if sim + f64::EPSILON < self.cfg.min_similarity {
            return Answer::abstained(); // nearest case too dissimilar to adapt
        }

        // Adapt: the case's solution must recommend the goal's (predicate, object); transfer it onto
        // the target's subject. (We adapt a recorded solution, we do not restate a known fact.)
        let goal_action: Feature = (query.goal.predicate.clone(), query.goal.object.clone());
        if !case.solution.contains(&goal_action) {
            return Answer::abstained();
        }
        let already_known = query
            .context
            .iter()
            .any(|t| t.subject == query.goal.subject && *t == query.goal);
        if already_known {
            return Answer::abstained();
        }

        Answer {
            value: Some(query.goal.clone()),
            // Evidence — NEVER a Derivation. This is what keeps adapted solutions Heuristic.
            proof: ProofTrace::Evidence {
                confidence: sim,
                why: vec![format!(
                    "retrieved case '{}' (Jaccard {:.3} over {} target features); adapted solution {}={}",
                    case.name,
                    sim,
                    target.len(),
                    goal_action.0,
                    goal_action.1
                )],
            },
            provenance: vec![format!("case-based:{}", case.name)],
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    /// `Substrate::Symbolic`: retrieve-and-adapt is an *algorithm* over feature sets (no neural net)
    /// — matching the `nusy-inductive` / `nusy-analogical` precedent for a symbolic-but-non-proof
    /// process. Its non-provability is carried by the `Evidence` proof + `sound: false`, **not** by
    /// the substrate tag.
    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }

    /// **Unsound and incomplete by nature** — the retrieved precedent can be wrong for the target
    /// (unsound) and the library may lack a relevant case (incomplete); answers carry a retrieval
    /// confidence (probabilistic). This is exactly why its answers can never be `Proven`.
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

    fn q(goal: Triple, context: Vec<Triple>) -> Query {
        Query { goal, context }
    }

    /// A clinical case library: two prior patients with their worked-up tests.
    fn clinical_library() -> Vec<Case> {
        vec![
            Case::new(
                "flu-case",
                vec![("has_symptom", "fever"), ("has_symptom", "cough")],
                vec![("order", "flu_test")],
            ),
            Case::new(
                "migraine-case",
                vec![("has_symptom", "headache"), ("has_symptom", "photophobia")],
                vec![("order", "neuro_referral")],
            ),
        ]
    }

    // ── Core: retrieve + adapt ──────────────────────────────────────────────

    #[test]
    fn retrieves_and_adapts_matching_case() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        let a = r.answer(&q(
            t("patient_b", "order", "flu_test"),
            vec![
                t("patient_b", "has_symptom", "fever"),
                t("patient_b", "has_symptom", "cough"),
            ],
        ));
        assert_eq!(a.value, Some(t("patient_b", "order", "flu_test")));
        assert_eq!(a.provability(), Provability::Heuristic);
        // Exact feature match → Jaccard 1.0.
        if let ProofTrace::Evidence { confidence, .. } = a.proof {
            assert!((confidence - 1.0).abs() < 1e-9);
        } else {
            panic!("expected an Evidence trace");
        }
        assert_eq!(a.provenance, vec!["case-based:flu-case".to_string()]);
    }

    #[test]
    fn partial_match_still_retrieves_above_threshold() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        // Target shares fever + cough with flu-case but adds one extra feature → Jaccard 2/3 > 0.5.
        let a = r.answer(&q(
            t("patient_c", "order", "flu_test"),
            vec![
                t("patient_c", "has_symptom", "fever"),
                t("patient_c", "has_symptom", "cough"),
                t("patient_c", "has_symptom", "fatigue"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Heuristic);
        if let ProofTrace::Evidence { confidence, .. } = a.proof {
            assert!((confidence - 2.0 / 3.0).abs() < 1e-9);
        } else {
            panic!("expected Evidence");
        }
    }

    // ── The guarantee invariant (load-bearing) ──────────────────────────────

    #[test]
    fn adapted_answer_is_heuristic_never_proven() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        let a = r.answer(&q(
            t("patient_b", "order", "flu_test"),
            vec![
                t("patient_b", "has_symptom", "fever"),
                t("patient_b", "has_symptom", "cough"),
            ],
        ));
        // THE invariant: a retrieved precedent is never a proof.
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_ne!(a.provability(), Provability::Proven);
        assert!(!r.guarantee().sound);
    }

    // ── Abstention ──────────────────────────────────────────────────────────

    #[test]
    fn abstains_when_no_case_is_similar() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        // Disjoint features from every case → no retrieval.
        let a = r.answer(&q(
            t("patient_d", "order", "flu_test"),
            vec![t("patient_d", "has_symptom", "rash")],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
        assert!(a.value.is_none());
    }

    #[test]
    fn abstains_when_nearest_case_solution_lacks_the_goal() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        // Retrieves flu-case (fever+cough) but the goal asks for a test that case never ordered.
        let a = r.answer(&q(
            t("patient_b", "order", "mri"),
            vec![
                t("patient_b", "has_symptom", "fever"),
                t("patient_b", "has_symptom", "cough"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn abstains_when_no_target_features() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        let a = r.answer(&Query::new(t("patient_b", "order", "flu_test")));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn abstains_when_goal_already_known() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        let a = r.answer(&q(
            t("patient_b", "order", "flu_test"),
            vec![
                t("patient_b", "has_symptom", "fever"),
                t("patient_b", "has_symptom", "cough"),
                t("patient_b", "order", "flu_test"), // already done
            ],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    // ── Nearest-case selection ──────────────────────────────────────────────

    #[test]
    fn retrieves_the_nearest_of_several_cases() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        // Closer to migraine-case (headache+photophobia) than flu-case.
        let a = r.answer(&q(
            t("patient_e", "order", "neuro_referral"),
            vec![
                t("patient_e", "has_symptom", "headache"),
                t("patient_e", "has_symptom", "photophobia"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_eq!(a.provenance, vec!["case-based:migraine-case".to_string()]);
    }

    // ── Genericity: the same engine over an incident-pattern playbook ────────

    #[test]
    fn generic_over_incident_playbook() {
        // No domain literals in the engine — an ops incident library works unchanged.
        let playbook = vec![
            Case::new(
                "lb-timeout",
                vec![("symptom", "timeout"), ("layer", "load_balancer")],
                vec![("remediate", "restart_lb")],
            ),
            Case::new(
                "db-deadlock",
                vec![("symptom", "deadlock"), ("layer", "database")],
                vec![("remediate", "kill_blocking_txn")],
            ),
        ];
        let r = CaseBasedReasoner::new(playbook, CbrConfig::default());
        let a = r.answer(&q(
            t("incident_42", "remediate", "restart_lb"),
            vec![
                t("incident_42", "symptom", "timeout"),
                t("incident_42", "layer", "load_balancer"),
            ],
        ));
        assert_eq!(a.value, Some(t("incident_42", "remediate", "restart_lb")));
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_eq!(a.provenance, vec!["case-based:lb-timeout".to_string()]);
    }

    // ── Threshold + determinism ─────────────────────────────────────────────

    #[test]
    fn respects_min_similarity_threshold() {
        // Strict config: require near-exact match. A 2/3 partial match is then rejected.
        let cfg = CbrConfig { min_similarity: 0.9 };
        let r = CaseBasedReasoner::new(clinical_library(), cfg);
        let a = r.answer(&q(
            t("patient_c", "order", "flu_test"),
            vec![
                t("patient_c", "has_symptom", "fever"),
                t("patient_c", "has_symptom", "cough"),
                t("patient_c", "has_symptom", "fatigue"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn cbr_is_deterministic() {
        let r = CaseBasedReasoner::new(clinical_library(), CbrConfig::default());
        let query = q(
            t("patient_b", "order", "flu_test"),
            vec![
                t("patient_b", "has_symptom", "fever"),
                t("patient_b", "has_symptom", "cough"),
            ],
        );
        let a = r.answer(&query);
        let b = r.answer(&query);
        assert_eq!(a.value, b.value);
        assert_eq!(a.provenance, b.provenance);
    }

    #[test]
    fn jaccard_basic_properties() {
        let s = |fs: &[(&str, &str)]| -> BTreeSet<Feature> {
            fs.iter().map(|(p, o)| (p.to_string(), o.to_string())).collect()
        };
        let a = s(&[("x", "1"), ("y", "2")]);
        let b = s(&[("x", "1"), ("y", "2")]);
        assert!((jaccard(&a, &b) - 1.0).abs() < 1e-9); // identical
        let c = s(&[("z", "9")]);
        assert!(jaccard(&a, &c).abs() < 1e-9); // disjoint
        let d = s(&[("x", "1")]);
        assert!((jaccard(&a, &d) - 0.5).abs() < 1e-9); // 1 shared / 2 union
    }
}
