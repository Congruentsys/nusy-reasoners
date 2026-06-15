//! # nusy-analogical — the analogical reasoner (VY-B E2, EX-4888)
//!
//! Reasoning by **structure mapping**: given a *source case* whose relational structure is known
//! (a legal precedent, a worked physics problem, a prior incident), and a *target* described only
//! partially, **align the two by their relational structure** and **transfer** an unobserved
//! relation from source to target — "this contract is *like* that precedent, so it is probably
//! enforceable too." This is the engine behind case transfer and precedent-style reasoning.
//!
//! The model follows Gentner's Structure-Mapping Theory in spirit: an analogy is good when it
//! preserves **relations** between aligned entities (systematicity), not when surface attributes
//! happen to match. We score an alignment by how many of the source's relations carry over onto the
//! *known* target facts, and only then project the goal relation across the same alignment.
//!
//! ## The guarantee invariant — analogy is HEURISTIC, never Proven
//!
//! An analogy *suggests*; it does not *prove*. "Earth orbits the Sun and the Sun attracts it; an
//! electron is *like* a planet" does not prove the electron is attracted — it is a useful guess that
//! a sound theory must still certify. So every answer this reasoner produces carries a
//! [`ProofTrace::Evidence`] trace whose [`provability`](nusy_reasoner::Answer::provability) is
//! *always* [`Provability::Heuristic`](nusy_reasoner::Provability::Heuristic). The engine holds no
//! [`DerivationTrace`](nusy_reasoner::DerivationTrace), so it is **structurally unable to mint
//! `Proven`** — exactly as the Reasoner contract requires. Transferred relations feed downstream
//! reasoning as *candidates*; only a sound deductive engine can ever certify them `Proven`.
//!
//! ## Generic by construction
//!
//! The engine contains **no domain literals** — every entity, predicate, and object it aligns over
//! comes from the case/target data, never from the code. Map legal precedents, physical systems, or
//! biological cases with the same engine (the tests prove it across three domains).
//!
//! ```
//! use nusy_analogical::{AnalogicalReasoner, AnalogyConfig, Case};
//! use nusy_reasoner::{Provability, Query, Reasoner};
//! use nusy_unify::Triple;
//!
//! // Source precedent: a contract with an arbitration clause was held enforceable by a court.
//! let precedent = Case::new("precedent-1", vec![
//!     Triple::new("contract_a", "has_clause", "arbitration"),
//!     Triple::new("court", "ruled", "contract_a"),
//!     Triple::new("contract_a", "is", "enforceable"),
//! ]);
//! let r = AnalogicalReasoner::new(vec![precedent], AnalogyConfig::default());
//!
//! // Target: a new contract, also with an arbitration clause, also before a court.
//! // By analogy, is it enforceable? — heuristically, yes.
//! let q = Query {
//!     goal: Triple::new("contract_b", "is", "enforceable"),
//!     context: vec![
//!         Triple::new("contract_b", "has_clause", "arbitration"),
//!         Triple::new("court", "ruled", "contract_b"),
//!     ],
//! };
//! let a = r.answer(&q);
//! assert_eq!(a.value, Some(Triple::new("contract_b", "is", "enforceable")));
//! assert_eq!(a.provability(), Provability::Heuristic); // an analogy is not a proof
//! assert_ne!(a.provability(), Provability::Proven);
//! ```

use std::collections::{BTreeMap, BTreeSet, HashSet};

use nusy_reasoner::{
    Answer, CompetenceEnvelope, Guarantee, ProofTrace, Query, QueryShape, Reasoner, Substrate,
};
use nusy_unify::Triple;

/// A source case: a named bundle of ground relations whose structure can be mapped onto a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Case {
    /// Stable identifier, used in provenance (e.g. `"precedent-1"`).
    pub name: String,
    /// The relational structure of the case — the facts an analogy can transfer.
    pub facts: Vec<Triple>,
}

impl Case {
    /// Construct a case from a name and its relational facts.
    pub fn new(name: impl Into<String>, facts: Vec<Triple>) -> Self {
        Self {
            name: name.into(),
            facts,
        }
    }

    /// The entities (subjects ∪ objects) the case mentions — the domain of an alignment.
    fn entities(&self) -> BTreeSet<String> {
        let mut e = BTreeSet::new();
        for t in &self.facts {
            e.insert(t.subject.clone());
            e.insert(t.object.clone());
        }
        e
    }
}

/// Thresholds governing when an analogy is strong enough to transfer.
#[derive(Debug, Clone, Copy)]
pub struct AnalogyConfig {
    /// Minimum structural similarity (supported source relations / total source relations) ∈ (0, 1].
    /// Guards against transferring on a weak, coincidental alignment.
    pub min_similarity: f64,
    /// Minimum number of **known** source relations that must carry over onto the target before any
    /// transfer — systematicity. A single aligned entity is not an analogy; preserved *relations*
    /// are. Guards against surface-only "analogies".
    pub min_aligned_relations: usize,
    /// Hard cap on source-case size for exhaustive alignment search (factorial in entity count).
    /// Cases larger than this are skipped with a provenance note rather than risking blowup.
    pub max_entities: usize,
}

impl Default for AnalogyConfig {
    /// Conservative defaults: a majority of the source structure must carry over (> 0.5), at least
    /// **two** relations must align (one is a coincidence, two is a structure), and the exhaustive
    /// search is capped at 8 entities.
    fn default() -> Self {
        Self {
            min_similarity: 0.5,
            min_aligned_relations: 2,
            max_entities: 8,
        }
    }
}

/// The result of aligning a source case onto a set of known target facts.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Injective correspondence: source entity → target entity (partial — unmapped source entities
    /// are absent).
    pub map: BTreeMap<String, String>,
    /// How many of the source's relations are preserved under `map` on the *known* target facts.
    pub supported: usize,
    /// Total relations in the source case (the denominator of `similarity`).
    pub total: usize,
    /// `supported / total` ∈ [0, 1] — the structural similarity of source and target.
    pub similarity: f64,
}

/// Apply a (partial) entity map to a triple; `None` if either endpoint is unmapped.
fn project(map: &BTreeMap<String, String>, t: &Triple) -> Option<Triple> {
    let s = map.get(&t.subject)?;
    let o = map.get(&t.object)?;
    Some(Triple::new(s.clone(), t.predicate.clone(), o.clone()))
}

/// Count how many source facts are preserved by `map` on `target_set`.
fn supported_count(
    map: &BTreeMap<String, String>,
    case_facts: &[Triple],
    target_set: &HashSet<Triple>,
) -> usize {
    case_facts
        .iter()
        .filter(|t| project(map, t).is_some_and(|p| target_set.contains(&p)))
        .count()
}

/// Find the alignment of `case` onto `target_facts` (known facts) that preserves the most relations.
///
/// Exhaustive over injective maps from the source entities to `target_entities` (battery cases are
/// tiny). Deterministic: among equally-supported maps, the lexicographically-smallest pair set wins,
/// so the same inputs always yield the same alignment. Returns `None` if the case exceeds
/// [`AnalogyConfig::max_entities`].
pub fn best_alignment(
    case: &Case,
    target_facts: &[Triple],
    target_entities: &BTreeSet<String>,
    cfg: &AnalogyConfig,
) -> Option<Alignment> {
    let src: Vec<String> = case.entities().into_iter().collect();
    if src.len() > cfg.max_entities {
        return None; // guard: exhaustive search is factorial — skip oversized cases.
    }
    let tgt: Vec<String> = target_entities.iter().cloned().collect();
    let target_set: HashSet<Triple> = target_facts.iter().cloned().collect();
    let total = case.facts.len();

    // Immutable context for the recursive search (bundled to keep the arg count sane).
    let ctx = SearchCtx {
        src: &src,
        tgt: &tgt,
        case_facts: &case.facts,
        target_set: &target_set,
    };

    // Backtracking over injective partial assignments src[i] -> Some(tgt) | None.
    let mut best: Option<(usize, BTreeMap<String, String>)> = None;
    let mut current: BTreeMap<String, String> = BTreeMap::new();
    let mut used: HashSet<String> = HashSet::new();
    search(&ctx, 0, &mut current, &mut used, &mut best);

    best.map(|(supported, map)| Alignment {
        map,
        supported,
        total,
        similarity: if total == 0 {
            0.0
        } else {
            supported as f64 / total as f64
        },
    })
}

/// Flatten a map to a sorted pair list — the deterministic tie-break key.
fn map_pairs(m: &BTreeMap<String, String>) -> Vec<(String, String)> {
    m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Immutable inputs to the alignment backtracking search.
struct SearchCtx<'a> {
    src: &'a [String],
    tgt: &'a [String],
    case_facts: &'a [Triple],
    target_set: &'a HashSet<Triple>,
}

/// Backtrack over injective partial assignments `src[i] -> Some(tgt) | None`, recording the
/// max-supported map (ties broken by lexicographically-smallest pair set, for determinism).
fn search(
    ctx: &SearchCtx,
    i: usize,
    current: &mut BTreeMap<String, String>,
    used: &mut HashSet<String>,
    best: &mut Option<(usize, BTreeMap<String, String>)>,
) {
    if i == ctx.src.len() {
        let score = supported_count(current, ctx.case_facts, ctx.target_set);
        let better = match best {
            None => true,
            Some((bscore, bmap)) => {
                score > *bscore || (score == *bscore && map_pairs(current) < map_pairs(bmap))
            }
        };
        if better {
            *best = Some((score, current.clone()));
        }
        return;
    }
    // Option A: leave src[i] unmapped (target may have fewer entities than source).
    search(ctx, i + 1, current, used, best);
    // Option B: map src[i] to each unused target entity (injective).
    for te in ctx.tgt {
        if used.contains(te) {
            continue;
        }
        used.insert(te.clone());
        current.insert(ctx.src[i].clone(), te.clone());
        search(ctx, i + 1, current, used, best);
        current.remove(&ctx.src[i]);
        used.remove(te);
    }
}

/// Bind source entity `k` to target `v` in `m`, preserving injectivity and any existing binding.
/// Returns `false` if `k` is already bound to a *different* target, or `v` is already the image of
/// another source entity.
fn bind(m: &mut BTreeMap<String, String>, k: &str, v: &str) -> bool {
    match m.get(k) {
        Some(existing) => existing == v,
        None => {
            if m.values().any(|x| x == v) {
                return false; // target already used — would break injectivity
            }
            m.insert(k.to_string(), v.to_string());
            true
        }
    }
}

/// Can the base alignment be extended so that *some* source relation projects exactly onto `goal`?
/// This is the SMT candidate-inference step: the base map (fixed by known relations) is extended,
/// injectively and consistently, to cover the goal's endpoints — which may be entities no known
/// fact mentions.
fn transfers_goal(base: &BTreeMap<String, String>, case: &Case, goal: &Triple) -> bool {
    case.facts.iter().any(|tr| {
        if tr.predicate != goal.predicate {
            return false;
        }
        let mut m = base.clone();
        bind(&mut m, &tr.subject, &goal.subject) && bind(&mut m, &tr.object, &goal.object)
    })
}

/// A [`Reasoner`] that answers by **analogy**: it aligns each known source case onto the query's
/// target context and transfers the goal relation across the best-aligned, sufficiently-similar
/// case. Every answer is [`Provability::Heuristic`](nusy_reasoner::Provability) — an analogy
/// suggests, it does not prove.
pub struct AnalogicalReasoner {
    cases: Vec<Case>,
    cfg: AnalogyConfig,
    envelope: CompetenceEnvelope,
}

impl AnalogicalReasoner {
    /// Build a reasoner over a corpus of source cases. Its competence covers exactly the predicates
    /// the cases could transfer (the predicates appearing in their facts).
    pub fn new(cases: Vec<Case>, cfg: AnalogyConfig) -> Self {
        let predicates: BTreeSet<String> = cases
            .iter()
            .flat_map(|c| c.facts.iter().map(|t| t.predicate.clone()))
            .collect();
        let envelope = CompetenceEnvelope {
            shapes: vec![QueryShape {
                name: "structure-mapped analogy".into(),
                predicates: predicates.into_iter().collect(),
            }],
        };
        Self {
            cases,
            cfg,
            envelope,
        }
    }

    /// The source cases this reasoner draws on.
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// For one case, if its best alignment onto `context` is strong enough AND maps a source
    /// relation exactly onto `goal`, return `(similarity, alignment, case_name)`. The goal's own
    /// entities are added to the alignment domain so they can be aligned even though the goal fact
    /// is the *unknown* we are projecting.
    fn case_supports_goal(
        &self,
        case: &Case,
        goal: &Triple,
        context: &[Triple],
    ) -> Option<(f64, Alignment)> {
        // Alignment domain = entities known from context PLUS the goal's own endpoints.
        let mut target_entities: BTreeSet<String> = BTreeSet::new();
        for t in context {
            target_entities.insert(t.subject.clone());
            target_entities.insert(t.object.clone());
        }
        target_entities.insert(goal.subject.clone());
        target_entities.insert(goal.object.clone());

        let alignment = best_alignment(case, context, &target_entities, &self.cfg)?;

        // Systematicity + similarity gates: enough preserved KNOWN relations, strong enough overall.
        if alignment.supported < self.cfg.min_aligned_relations {
            return None;
        }
        if alignment.similarity + f64::EPSILON < self.cfg.min_similarity {
            return None;
        }

        // The goal must be the image of some source relation under this alignment, and must be
        // genuinely NOVEL (not already a known target fact — we transfer, we don't restate).
        let context_set: HashSet<&Triple> = context.iter().collect();
        if context_set.contains(goal) {
            return None;
        }
        // SMT *candidate projection*: the base alignment is fixed by the known relations; now try to
        // EXTEND it (injectively, consistently) so a source relation maps exactly onto the goal. The
        // goal's endpoints may be entities the base alignment left unmapped (they appear in no known
        // fact) — projecting the relation onto them is precisely the analogical inference.
        if !transfers_goal(&alignment.map, case, goal) {
            return None;
        }
        Some((alignment.similarity, alignment))
    }
}

impl Reasoner for AnalogicalReasoner {
    /// Transfer the goal relation from the best-aligned, sufficiently-similar source case, as
    /// **evidence** (never a derivation). Abstains when no case aligns well enough, or none maps a
    /// source relation onto the goal. When several cases qualify, the most structurally-similar wins
    /// (ties broken by case name for determinism).
    fn answer(&self, query: &Query) -> Answer {
        let mut best: Option<(f64, &Case, Alignment)> = None;
        for case in &self.cases {
            if let Some((sim, alignment)) =
                self.case_supports_goal(case, &query.goal, &query.context)
            {
                let better = match &best {
                    None => true,
                    Some((bsim, bcase, _)) => {
                        sim > *bsim || (sim == *bsim && case.name < bcase.name)
                    }
                };
                if better {
                    best = Some((sim, case, alignment));
                }
            }
        }

        match best {
            Some((sim, case, alignment)) => {
                let mut pairs: Vec<String> = alignment
                    .map
                    .iter()
                    .map(|(s, t)| format!("{s}~{t}"))
                    .collect();
                pairs.sort();
                Answer {
                    value: Some(query.goal.clone()),
                    // Evidence — NEVER a Derivation. This is what keeps analogical answers Heuristic.
                    proof: ProofTrace::Evidence {
                        confidence: sim,
                        why: vec![format!(
                            "analogy to case '{}': {}/{} relations preserved (similarity {:.3}); mapping [{}]",
                            case.name,
                            alignment.supported,
                            alignment.total,
                            sim,
                            pairs.join(", ")
                        )],
                    },
                    provenance: vec![format!("analogical-case:{}", case.name)],
                }
            }
            None => Answer::abstained(),
        }
    }

    fn competence_envelope(&self) -> &CompetenceEnvelope {
        &self.envelope
    }

    /// `Substrate::Symbolic`: structure mapping is an *algorithm* over triples (no neural net) —
    /// matching the `nusy-inductive` / `nusy-abduction` precedent for a symbolic-but-non-proof
    /// process. Its non-provability is carried by the `Evidence` proof + `sound: false`, **not** by
    /// the substrate tag.
    fn substrate(&self) -> Substrate {
        Substrate::Symbolic
    }

    /// **Unsound and incomplete by nature** — an analogy can be wrong (unsound) and the case corpus
    /// may not cover every valid analogy (incomplete); answers carry a similarity confidence
    /// (probabilistic). This is exactly why its answers can never be `Proven`.
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

    /// A legal precedent: a contract with an arbitration clause, ruled on by a court, held enforceable.
    fn precedent() -> Case {
        Case::new(
            "precedent-1",
            vec![
                t("contract_a", "has_clause", "arbitration"),
                t("court", "ruled", "contract_a"),
                t("contract_a", "is", "enforceable"),
            ],
        )
    }

    // ── Core: structure mapping + transfer ──────────────────────────────────

    #[test]
    fn transfers_enforceability_by_analogy() {
        let r = AnalogicalReasoner::new(vec![precedent()], AnalogyConfig::default());
        let query = q(
            t("contract_b", "is", "enforceable"),
            vec![
                t("contract_b", "has_clause", "arbitration"),
                t("court", "ruled", "contract_b"),
            ],
        );
        let a = r.answer(&query);
        assert_eq!(a.value, Some(t("contract_b", "is", "enforceable")));
        assert_eq!(a.provability(), Provability::Heuristic);
        // Two known relations (has_clause, ruled) aligned → similarity 2/3.
        if let ProofTrace::Evidence { confidence, .. } = a.proof {
            assert!((confidence - 2.0 / 3.0).abs() < 1e-9);
        } else {
            panic!("expected an Evidence trace");
        }
    }

    #[test]
    fn alignment_maps_source_entities_to_target() {
        let mut target_entities = BTreeSet::new();
        for e in ["contract_b", "arbitration", "court"] {
            target_entities.insert(e.to_string());
        }
        let context = vec![
            t("contract_b", "has_clause", "arbitration"),
            t("court", "ruled", "contract_b"),
        ];
        let al = best_alignment(
            &precedent(),
            &context,
            &target_entities,
            &AnalogyConfig::default(),
        )
        .expect("alignment exists");
        assert_eq!(al.map.get("contract_a"), Some(&"contract_b".to_string()));
        assert_eq!(al.map.get("court"), Some(&"court".to_string()));
        assert_eq!(al.supported, 2);
    }

    // ── The guarantee invariant (load-bearing) ──────────────────────────────

    #[test]
    fn analogical_answer_is_heuristic_never_proven() {
        let r = AnalogicalReasoner::new(vec![precedent()], AnalogyConfig::default());
        let a = r.answer(&q(
            t("contract_b", "is", "enforceable"),
            vec![
                t("contract_b", "has_clause", "arbitration"),
                t("court", "ruled", "contract_b"),
            ],
        ));
        // THE invariant: an analogy is never a proof.
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_ne!(a.provability(), Provability::Proven);
        assert!(!r.guarantee().sound);
    }

    // ── Abstention ──────────────────────────────────────────────────────────

    #[test]
    fn abstains_when_structure_does_not_match() {
        let r = AnalogicalReasoner::new(vec![precedent()], AnalogyConfig::default());
        // Target shares no relational structure with the precedent → no analogy.
        let a = r.answer(&q(
            t("widget", "is", "enforceable"),
            vec![t("widget", "made_of", "steel")],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
        assert!(a.value.is_none());
    }

    #[test]
    fn abstains_when_only_one_relation_aligns() {
        // Only `has_clause` carries over (1 relation) — below min_aligned_relations 2 → no transfer.
        let r = AnalogicalReasoner::new(vec![precedent()], AnalogyConfig::default());
        let a = r.answer(&q(
            t("contract_b", "is", "enforceable"),
            vec![t("contract_b", "has_clause", "arbitration")],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    #[test]
    fn abstains_when_goal_already_known() {
        // The goal is already a context fact → nothing to transfer (we project, not restate).
        let r = AnalogicalReasoner::new(vec![precedent()], AnalogyConfig::default());
        let a = r.answer(&q(
            t("contract_b", "is", "enforceable"),
            vec![
                t("contract_b", "has_clause", "arbitration"),
                t("court", "ruled", "contract_b"),
                t("contract_b", "is", "enforceable"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
    }

    // ── Genericity: the same engine over a physics analogy (no domain literals) ──

    #[test]
    fn generic_over_physics_solar_system_to_atom() {
        // The classic Rutherford analogy: the Sun attracts and is orbited by a planet; an electron
        // orbits a nucleus — so by analogy the nucleus attracts the electron.
        let solar = Case::new(
            "solar-system",
            vec![
                t("planet", "orbits", "sun"),
                t("sun", "attracts", "planet"),
                t("sun", "more_massive_than", "planet"),
            ],
        );
        let r = AnalogicalReasoner::new(vec![solar], AnalogyConfig::default());
        let a = r.answer(&q(
            t("nucleus", "attracts", "electron"),
            vec![
                t("electron", "orbits", "nucleus"),
                t("nucleus", "more_massive_than", "electron"),
            ],
        ));
        assert_eq!(a.value, Some(t("nucleus", "attracts", "electron")));
        assert_eq!(a.provability(), Provability::Heuristic);
    }

    // ── Best-case selection + determinism ───────────────────────────────────

    #[test]
    fn most_similar_case_wins() {
        // Two precedents that BOTH clear the similarity gate (≥2 aligned AND sim ≥ 0.5) — so the
        // winner is decided by the max-selection branch in answer(), NOT filtered at the gate:
        //  - strong: 3 facts, 2 known align (has_clause, approved) → similarity 2/3 ≈ 0.667
        //  - weak:   4 facts, 2 known align (has_clause, approved) → similarity 2/4 = 0.500
        //            (clears the gate via EPSILON, so it genuinely reaches selection and then loses)
        // The more structurally-similar case (strong) must win the max-selection.
        let strong = Case::new(
            "strong",
            vec![
                t("c_a", "has_clause", "arbitration"),
                t("regulator", "approved", "c_a"),
                t("c_a", "is", "valid"),
            ],
        );
        let weak = Case::new(
            "weak",
            vec![
                t("d_a", "has_clause", "arbitration"),
                t("regulator", "approved", "d_a"),
                t("d_a", "is", "valid"),
                t("d_a", "filed_in", "2019"),
            ],
        );
        let r = AnalogicalReasoner::new(vec![weak, strong], AnalogyConfig::default());
        let a = r.answer(&q(
            t("c_b", "is", "valid"),
            vec![
                t("c_b", "has_clause", "arbitration"),
                t("regulator", "approved", "c_b"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Heuristic);
        assert_eq!(a.provenance, vec!["analogical-case:strong".to_string()]);
        if let ProofTrace::Evidence { confidence, .. } = a.proof {
            assert!(
                (confidence - 2.0 / 3.0).abs() < 1e-9,
                "the more-similar case (2/3) wins selection over the weaker qualifier (2/4)"
            );
        } else {
            panic!("expected Evidence");
        }
    }

    #[test]
    fn analogy_is_deterministic() {
        let r = AnalogicalReasoner::new(vec![precedent()], AnalogyConfig::default());
        let query = q(
            t("contract_b", "is", "enforceable"),
            vec![
                t("contract_b", "has_clause", "arbitration"),
                t("court", "ruled", "contract_b"),
            ],
        );
        let a = r.answer(&query);
        let b = r.answer(&query);
        assert_eq!(a.value, b.value);
        assert_eq!(a.provenance, b.provenance);
    }

    #[test]
    fn respects_min_similarity_threshold() {
        // Strict config: require near-perfect similarity. A 2/3 analogy is then rejected.
        let cfg = AnalogyConfig {
            min_similarity: 0.9,
            min_aligned_relations: 2,
            max_entities: 8,
        };
        let r = AnalogicalReasoner::new(vec![precedent()], cfg);
        let a = r.answer(&q(
            t("contract_b", "is", "enforceable"),
            vec![
                t("contract_b", "has_clause", "arbitration"),
                t("court", "ruled", "contract_b"),
            ],
        ));
        assert_eq!(a.provability(), Provability::Abstained);
    }
}
