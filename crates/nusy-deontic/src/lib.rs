//! # nusy-deontic — obligations as first-class, compliance as proof-carrying derivation (EX-4781, VY-F E2)
//!
//! A guideline doesn't only say *what follows* — it says what you **must**, **may**, or **must not**
//! do, and what you may skip **if you document why**. This crate makes those four deontic modalities
//! first-class [`Obligation`]s and turns conformance into a **proof-carrying** [`check_compliance`]
//! pass whose every verdict cites the obligation's `source_artifact`.
//!
//! ## Generic by construction
//!
//! Per EX-4781's constraints, the core is **shape-generic** (RULES+DAG rows, never clinical structs)
//! and has **no `arrow-core` dependency**: obligations are extracted from a minimal [`PlanActionRow`]
//! (the FHIR-CPG `requiredBehavior` / `doNotPerform` surface, generalized) via
//! [`obligations_from_rows`], and artifact identity is just a `String` the caller supplies — no
//! lookups into a concrete store. The modality vocabulary is **fixed to the four** below (open vocab
//! deferred), mirroring FHIR R4 `ActionRequiredBehavior` + `doNotPerform`.
//!
//! ## The four modalities → compliance semantics
//!
//! For each obligation whose `condition` is active (a `None` condition is always active):
//! - [`Must`](Modality::Must): the action **must** be proposed. Absent ⇒ [`Violation`](Compliance::Violation).
//! - [`Could`](Modality::Could): permitted; never a violation.
//! - [`MustUnlessDocumented`](Modality::MustUnlessDocumented): like `Must`, **but** an absent action
//!   with a *documented* exception is compliant; absent **and** undocumented ⇒
//!   [`NeedsDocumentation`](Compliance::NeedsDocumentation) — the escape hatch, not a hard violation.
//! - [`DoNotPerform`](Modality::DoNotPerform): the action **must not** be proposed. Present ⇒
//!   [`Violation`](Compliance::Violation) (EX-4692 explicit negative, as an obligation).
//!
//! ## Behind the reasoner contract
//!
//! [`DeonticReasoner`] implements [`Reasoner`]: a goal `(action, "compliant_with", artifact)` is
//! answered **`Proven`** (with a derivation citing the source artifact) iff *every* applicable
//! obligation on that action is compliant given the context facts; otherwise it **abstains** — the
//! detailed verdicts (violations, needs-documentation) are surfaced by [`check_compliance`], which is
//! the richer, deontic-native API. Compliance is sound: a `Proven` answer means no obligation is
//! violated and none needs documentation.

use std::collections::HashSet;

use nusy_reasoner::{
    Answer, CompetenceEnvelope, DerivationTrace, Guarantee, ProofTrace, Query, QueryShape,
    Reasoner, Substrate,
};
use nusy_unify::Triple;

/// FHIR R4 `requiredBehavior` code: a strong obligation.
pub const RB_MUST: &str = "must";
/// FHIR R4 `requiredBehavior` code: a permission.
pub const RB_COULD: &str = "could";
/// FHIR R4 `requiredBehavior` code: obligatory unless a documented exception exists.
pub const RB_MUST_UNLESS_DOCUMENTED: &str = "must-unless-documented";

/// The predicate the [`DeonticReasoner`] answers: `(action, "compliant_with", artifact)`.
pub const COMPLIANT_WITH: &str = "compliant_with";

/// The four deontic modalities (fixed vocabulary; open vocab deferred per EX-4781).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// The action must be performed (proposed) when applicable.
    Must,
    /// The action is permitted but not required.
    Could,
    /// Obligatory unless a documented exception is recorded.
    MustUnlessDocumented,
    /// The action must NOT be performed (a prohibition / explicit negative, EX-4692).
    DoNotPerform,
}

/// An obligation lifted from a guideline action: a modality, the action it governs, the artifact it
/// came from (for proof citation), and an optional applicability condition token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obligation {
    /// The deontic force.
    pub modality: Modality,
    /// The governed action id.
    pub action: String,
    /// The artifact this obligation is sourced from — cited in every verdict's proof.
    pub source_artifact: String,
    /// An opaque applicability-condition token; `None` = unconditional. The caller marks which
    /// condition tokens are active (deontic does not evaluate CQL — the engine resolves applicability).
    pub condition: Option<String>,
}

/// A minimal CPG-shaped plan row — the generalized `requiredBehavior` / `doNotPerform` surface,
/// decoupled from any clinical or Arrow type (EX-4781 genericity constraint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanActionRow {
    /// The action id.
    pub action_id: String,
    /// FHIR `requiredBehavior` code (`"must"` / `"could"` / `"must-unless-documented"`), if any.
    pub required_behavior: Option<String>,
    /// FHIR `doNotPerform` — `true` makes this action a prohibition (takes precedence over behavior).
    pub do_not_perform: bool,
    /// Optional applicability-condition token.
    pub condition: Option<String>,
    /// The artifact the row belongs to.
    pub source_artifact: String,
}

/// Map CPG-shaped rows to obligations. `doNotPerform` takes precedence (a prohibition is a
/// prohibition regardless of `requiredBehavior`); otherwise the `requiredBehavior` code maps to a
/// modality, and a row with neither a recognized behavior nor `doNotPerform` is treated as a plain
/// permission ([`Could`](Modality::Could)) — a recommendation you *may* follow.
pub fn obligations_from_rows(rows: &[PlanActionRow]) -> Vec<Obligation> {
    rows.iter()
        .map(|r| {
            let modality = if r.do_not_perform {
                Modality::DoNotPerform
            } else {
                match r.required_behavior.as_deref() {
                    Some(RB_MUST) => Modality::Must,
                    Some(RB_MUST_UNLESS_DOCUMENTED) => Modality::MustUnlessDocumented,
                    // "could", an unrecognized code, or absence ⇒ a permission.
                    _ => Modality::Could,
                }
            };
            Obligation {
                modality,
                action: r.action_id.clone(),
                source_artifact: r.source_artifact.clone(),
                condition: r.condition.clone(),
            }
        })
        .collect()
}

/// The compliance status of one obligation against a proposed action set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compliance {
    /// The obligation is satisfied.
    Compliant,
    /// The obligation is violated (a `Must` unmet, or a `DoNotPerform` action proposed).
    Violation,
    /// A `MustUnlessDocumented` action is absent and no exception is documented — needs documentation.
    NeedsDocumentation,
}

/// A proof-carrying verdict: the obligation, its compliance status, and evidence citing the source
/// artifact (the "why" of the verdict).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// The obligation evaluated.
    pub obligation: Obligation,
    /// Its compliance status.
    pub compliance: Compliance,
    /// Evidence chain — cites the obligation's source artifact and the rule applied.
    pub evidence: Vec<String>,
}

/// The world the obligations are checked against: which actions were proposed, which carry a
/// documented exception, and which condition tokens are active.
#[derive(Debug, Clone, Default)]
pub struct ComplianceInput {
    /// Action ids that were proposed/performed.
    pub proposed: HashSet<String>,
    /// Action ids with a documented exception (satisfies a `MustUnlessDocumented` skip).
    pub documented: HashSet<String>,
    /// Active applicability-condition tokens (an obligation with `None` condition is always active).
    pub active_conditions: HashSet<String>,
}

impl ComplianceInput {
    /// Is this obligation applicable now? (`None` condition ⇒ always; else the token must be active.)
    fn applies(&self, ob: &Obligation) -> bool {
        match &ob.condition {
            None => true,
            Some(c) => self.active_conditions.contains(c),
        }
    }
}

/// Check a set of obligations against the proposed action set. Returns one [`Verdict`] per
/// **applicable** obligation (inapplicable obligations are omitted — they impose nothing now).
/// Every verdict is proof-carrying: its `evidence` cites the obligation's `source_artifact`.
pub fn check_compliance(obligations: &[Obligation], input: &ComplianceInput) -> Vec<Verdict> {
    obligations
        .iter()
        .filter(|ob| input.applies(ob))
        .map(|ob| {
            let present = input.proposed.contains(&ob.action);
            let documented = input.documented.contains(&ob.action);
            let (compliance, rule) = match ob.modality {
                Modality::Must => {
                    if present {
                        (Compliance::Compliant, "must:proposed")
                    } else {
                        (Compliance::Violation, "must:absent")
                    }
                }
                Modality::Could => (Compliance::Compliant, "could:permitted"),
                Modality::MustUnlessDocumented => {
                    if present {
                        (Compliance::Compliant, "must-unless-documented:proposed")
                    } else if documented {
                        (
                            Compliance::Compliant,
                            "must-unless-documented:documented-exception",
                        )
                    } else {
                        (
                            Compliance::NeedsDocumentation,
                            "must-unless-documented:absent-undocumented",
                        )
                    }
                }
                Modality::DoNotPerform => {
                    if present {
                        (Compliance::Violation, "do-not-perform:proposed")
                    } else {
                        (Compliance::Compliant, "do-not-perform:absent")
                    }
                }
            };
            Verdict {
                obligation: ob.clone(),
                compliance,
                evidence: vec![
                    format!("artifact:{}", ob.source_artifact),
                    format!("rule:{rule}"),
                    format!("action:{}", ob.action),
                ],
            }
        })
        .collect()
}

/// A deontic reasoner: a set of obligations evaluated as proof-carrying compliance behind the
/// [`Reasoner`] contract.
#[derive(Debug, Clone)]
pub struct DeonticReasoner {
    obligations: Vec<Obligation>,
    envelope: CompetenceEnvelope,
}

impl DeonticReasoner {
    /// Build a reasoner over a set of obligations.
    pub fn new(obligations: Vec<Obligation>) -> Self {
        let envelope = CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "deontic-compliance".to_string(),
                predicates: vec![COMPLIANT_WITH.to_string()],
            }],
        };
        Self {
            obligations,
            envelope,
        }
    }

    /// The obligations held by this reasoner.
    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }

    /// Full proof-carrying compliance report for an input world (the deontic-native API).
    pub fn check(&self, input: &ComplianceInput) -> Vec<Verdict> {
        check_compliance(&self.obligations, input)
    }
}

/// Read a [`ComplianceInput`] from context facts: `(action, "proposed", "true")`,
/// `(action, "documented", "true")`, `(condition, "active", "true")`.
fn input_from_context(context: &[Triple]) -> ComplianceInput {
    let mut input = ComplianceInput::default();
    for t in context {
        match t.predicate.as_str() {
            "proposed" if t.object == "true" => {
                input.proposed.insert(t.subject.clone());
            }
            "documented" if t.object == "true" => {
                input.documented.insert(t.subject.clone());
            }
            "active" if t.object == "true" => {
                input.active_conditions.insert(t.subject.clone());
            }
            _ => {}
        }
    }
    input
}

impl Reasoner for DeonticReasoner {
    fn answer(&self, query: &Query) -> Answer {
        let goal = &query.goal;
        if goal.predicate != COMPLIANT_WITH {
            return Answer::abstained();
        }
        // Goal: (action, "compliant_with", artifact). Prove the action complies with *every*
        // applicable obligation it carries from that artifact.
        let action = &goal.subject;
        let artifact = &goal.object;
        let input = input_from_context(&query.context);

        let relevant: Vec<&Obligation> = self
            .obligations
            .iter()
            .filter(|ob| {
                &ob.action == action && &ob.source_artifact == artifact && input.applies(ob)
            })
            .collect();

        if relevant.is_empty() {
            // No applicable obligation to prove compliance against — abstain (nothing to certify).
            return Answer::abstained();
        }

        let verdicts = check_compliance(
            &relevant.iter().map(|o| (*o).clone()).collect::<Vec<_>>(),
            &input,
        );
        let all_compliant = verdicts
            .iter()
            .all(|v| v.compliance == Compliance::Compliant);
        if !all_compliant {
            // A violation or needs-documentation — compliance is not Proven; abstain (detail via check()).
            return Answer::abstained();
        }

        // Proof: compliance holds BECAUSE every applicable obligation from the artifact is satisfied.
        let premises: Vec<DerivationTrace> = verdicts
            .iter()
            .map(|v| {
                DerivationTrace::Axiom(Triple::new(
                    v.obligation.action.clone(),
                    "obligation_satisfied",
                    format!("{:?}", v.obligation.modality),
                ))
            })
            .collect();
        let trace = DerivationTrace::Derived {
            conclusion: goal.clone(),
            rule_id: format!("deontic:compliant@{artifact}"),
            premises,
        };
        Answer {
            value: Some(goal.clone()),
            proof: ProofTrace::Derivation(trace),
            provenance: vec![format!("artifact:{artifact}")],
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

#[cfg(test)]
mod tests {
    use super::*;
    use nusy_reasoner::Provability;

    fn row(id: &str, rb: Option<&str>, dnp: bool) -> PlanActionRow {
        PlanActionRow {
            action_id: id.to_string(),
            required_behavior: rb.map(str::to_string),
            do_not_perform: dnp,
            condition: None,
            source_artifact: "guideline-x".to_string(),
        }
    }

    #[test]
    fn extraction_maps_required_behavior_and_do_not_perform() {
        let obs = obligations_from_rows(&[
            row("a", Some("must"), false),
            row("b", Some("could"), false),
            row("c", Some("must-unless-documented"), false),
            row("d", None, true),         // doNotPerform takes precedence
            row("e", Some("must"), true), // doNotPerform overrides a "must"
            row("f", None, false),        // bare recommendation ⇒ Could
        ]);
        let mods: Vec<Modality> = obs.iter().map(|o| o.modality).collect();
        assert_eq!(
            mods,
            vec![
                Modality::Must,
                Modality::Could,
                Modality::MustUnlessDocumented,
                Modality::DoNotPerform,
                Modality::DoNotPerform,
                Modality::Could,
            ]
        );
        // Source artifact is carried for proof citation.
        assert!(obs.iter().all(|o| o.source_artifact == "guideline-x"));
    }

    #[test]
    fn must_satisfied_and_violated() {
        let obs = obligations_from_rows(&[row("acei", Some("must"), false)]);
        // proposed ⇒ compliant
        let mut input = ComplianceInput::default();
        input.proposed.insert("acei".to_string());
        assert_eq!(
            check_compliance(&obs, &input)[0].compliance,
            Compliance::Compliant
        );
        // absent ⇒ violation
        let empty = ComplianceInput::default();
        let v = &check_compliance(&obs, &empty)[0];
        assert_eq!(v.compliance, Compliance::Violation);
        assert!(v.evidence.iter().any(|e| e == "artifact:guideline-x"));
    }

    #[test]
    fn do_not_perform_violation_when_prohibited_action_proposed() {
        let obs = obligations_from_rows(&[row("nsaid", None, true)]);
        let mut input = ComplianceInput::default();
        input.proposed.insert("nsaid".to_string());
        let v = &check_compliance(&obs, &input)[0];
        assert_eq!(v.compliance, Compliance::Violation);
        assert!(
            v.evidence
                .iter()
                .any(|e| e == "rule:do-not-perform:proposed")
        );
        // not proposed ⇒ compliant
        assert_eq!(
            check_compliance(&obs, &ComplianceInput::default())[0].compliance,
            Compliance::Compliant
        );
    }

    #[test]
    fn must_unless_documented_escape_hatch() {
        let obs = obligations_from_rows(&[row("statin", Some("must-unless-documented"), false)]);
        // present ⇒ compliant
        let mut present = ComplianceInput::default();
        present.proposed.insert("statin".to_string());
        assert_eq!(
            check_compliance(&obs, &present)[0].compliance,
            Compliance::Compliant
        );
        // absent + documented exception ⇒ compliant
        let mut documented = ComplianceInput::default();
        documented.documented.insert("statin".to_string());
        assert_eq!(
            check_compliance(&obs, &documented)[0].compliance,
            Compliance::Compliant
        );
        // absent + undocumented ⇒ needs documentation (NOT a hard violation)
        let absent = ComplianceInput::default();
        assert_eq!(
            check_compliance(&obs, &absent)[0].compliance,
            Compliance::NeedsDocumentation
        );
    }

    #[test]
    fn could_is_always_compliant() {
        let obs = obligations_from_rows(&[row("optional", Some("could"), false)]);
        assert_eq!(
            check_compliance(&obs, &ComplianceInput::default())[0].compliance,
            Compliance::Compliant
        );
    }

    #[test]
    fn inactive_condition_obligations_are_not_evaluated() {
        let obs = vec![Obligation {
            modality: Modality::Must,
            action: "dialysis".into(),
            source_artifact: "g".into(),
            condition: Some("esrd".into()),
        }];
        // condition not active ⇒ no verdict (imposes nothing now)
        let verdicts = check_compliance(&obs, &ComplianceInput::default());
        assert!(verdicts.is_empty());
        // condition active + action absent ⇒ a violation now appears
        let mut active = ComplianceInput::default();
        active.active_conditions.insert("esrd".into());
        assert_eq!(
            check_compliance(&obs, &active)[0].compliance,
            Compliance::Violation
        );
    }

    #[test]
    fn reasoner_proves_compliance_with_artifact_citation() {
        let r = DeonticReasoner::new(obligations_from_rows(&[row("acei", Some("must"), false)]));
        let ans = r.answer(&Query {
            goal: Triple::new("acei", COMPLIANT_WITH, "guideline-x"),
            context: vec![Triple::new("acei", "proposed", "true")],
        });
        assert_eq!(ans.provability(), Provability::Proven);
        assert_eq!(ans.provenance, vec!["artifact:guideline-x".to_string()]);
        if let ProofTrace::Derivation(DerivationTrace::Derived { rule_id, .. }) = &ans.proof {
            assert!(
                rule_id.contains("guideline-x"),
                "proof cites the source artifact"
            );
        } else {
            panic!("expected a derivation proof");
        }
    }

    #[test]
    fn reasoner_abstains_on_violation() {
        let r = DeonticReasoner::new(obligations_from_rows(&[row("acei", Some("must"), false)]));
        // 'acei' not proposed ⇒ Must violated ⇒ compliance not provable ⇒ abstain.
        let ans = r.answer(&Query {
            goal: Triple::new("acei", COMPLIANT_WITH, "guideline-x"),
            context: vec![],
        });
        assert_eq!(ans.provability(), Provability::Abstained);
        // The rich verdict is still available via the deontic-native API.
        let v = r.check(&ComplianceInput::default());
        assert_eq!(v[0].compliance, Compliance::Violation);
    }

    #[test]
    fn reasoner_abstains_on_do_not_perform_violation() {
        let r = DeonticReasoner::new(obligations_from_rows(&[row("nsaid", None, true)]));
        let ans = r.answer(&Query {
            goal: Triple::new("nsaid", COMPLIANT_WITH, "guideline-x"),
            context: vec![Triple::new("nsaid", "proposed", "true")], // prohibited action proposed
        });
        assert_eq!(ans.provability(), Provability::Abstained);
    }

    #[test]
    fn reasoner_covers_only_compliant_with() {
        let r = DeonticReasoner::new(vec![]);
        assert!(r.competence_envelope().covers(&Query {
            goal: Triple::new("a", COMPLIANT_WITH, "g"),
            context: vec![],
        }));
        assert!(!r.competence_envelope().covers(&Query {
            goal: Triple::new("a", "treats", "g"),
            context: vec![],
        }));
        assert_eq!(r.substrate(), Substrate::Symbolic);
        assert!(r.guarantee().sound);
    }
}
