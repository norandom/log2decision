//! Exact joins of existing keys to the originals of a scope (requirement 1.3; the
//! table "Native evidence | How to join it" of design.md).
//!
//! An artifact names its records in its producer's terms: a selection or a score part
//! by first position (for an archive the packed entry and line), a selection over a
//! store by store ID, a policy decision by the ID derived from the query's source
//! path and original text. Before a key is looked at, the source the artifact records
//! is compared with what the scope binds: another kind of source, other bytes, another
//! entry filter, another or an unrecorded reader interpretation and another source
//! location are each their own error. A key that names no original of the scope, and
//! two keys that name the same original, are errors too. Nothing is guessed.
//!
//! A join reads the relation of a committed scope, so it takes no host lease. The one
//! thing it keeps is a bit per original, held in the budget for the join or, when that
//! is not admitted, in the scope's work directory.

use std::fmt;

use l2d_core::selections::{Member, SelectionSet};
use l2d_core::{L2dError, MarkedInput, MarkedSource, MemoryBudget};
use l2d_decide::policy::{PolicyDecision, PolicyReport};

use super::{NativeId, Original, OriginalId, Scope, SourceKind};

/// What a join holds beside its bit per original: the identity and the original in
/// flight.
pub const JOIN_FIXED_BYTES: u64 = 4 * 1024;

/// A key of an existing artifact, in its producer's terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeKey<'a> {
    /// The first position of a record in a file or an archive: a selection member, a
    /// marked position, or the entry and line of a score part row packed with
    /// `ArchivePosition::pack`.
    Position(u64),
    /// The ID of a stored log.
    StoreId(&'a str),
}

/// What a completed join resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct JoinReport {
    /// The keys given.
    pub keys: u64,
    /// The distinct originals they name; equal to `keys`, since no two keys may name
    /// one original.
    pub originals: u64,
    /// The occurrences of those originals in the scope.
    pub occurrences: u64,
}

/// Why keys were not joined to a scope.
#[derive(Debug)]
pub enum JoinError {
    /// The artifact was made on another kind of source than the scope is over.
    SourceKind {
        artifact: &'static str,
        scope: &'static str,
    },
    /// The artifact was made on other bytes of the source, or other logs of the store.
    SourceChanged {
        artifact_sha256: String,
        scope_sha256: String,
    },
    /// The artifact was made with another archive entry filter.
    Filters {
        artifact: Vec<String>,
        scope: Vec<String>,
    },
    /// The artifact was made under another reader interpretation.
    Interpretation { detail: String },
    /// The artifact names another source location or store.
    Path { artifact: String, scope: String },
    /// The policy report derived its IDs from another path than the scope was opened
    /// at.
    PolicyPath { report: String, scope: String },
    /// The artifact does not record what its keys were made on.
    Unverified { detail: String },
    /// The producer of the artifact does not accept this kind of scope.
    UnsupportedProducer { detail: String },
    /// The key is of a kind this scope's source does not have.
    KeyKind { key: String, scope: &'static str },
    /// The key names no original of the scope.
    Unknown { key: String },
    /// A second key names an original that an earlier key named.
    Ambiguous { key: String, original: OriginalId },
    /// A record of the artifact contradicts the artifact itself.
    Drift { detail: String },
    /// Memory, disk or the callback failed.
    Failed(L2dError),
}

impl fmt::Display for JoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JoinError::SourceKind { artifact, scope } => write!(
                f,
                "the keys were made on a {artifact} and the scope is over a {scope}; nothing \
                 was joined"
            ),
            JoinError::SourceChanged {
                artifact_sha256,
                scope_sha256,
            } => write!(
                f,
                "the keys were made on content with the SHA-256 {artifact_sha256} and the scope \
                 holds {scope_sha256}: the source changed; make the artifact again on this \
                 revision or open the scope on the original one"
            ),
            JoinError::Filters { artifact, scope } => write!(
                f,
                "the keys were made with the entry patterns {artifact:?} and the scope was \
                 opened with {scope:?}; nothing was joined"
            ),
            JoinError::Interpretation { detail } => write!(
                f,
                "the keys were made under another reading of the source ({detail}); open the \
                 scope with the original options or make the artifact again"
            ),
            JoinError::Path { artifact, scope } => write!(
                f,
                "the keys were made on {artifact} and the scope is over {scope}: equal content \
                 at another location is another source; nothing was joined"
            ),
            JoinError::PolicyPath { report, scope } => write!(
                f,
                "the policy report derived its IDs from the source path {report} and the scope \
                 was opened at {scope}; this needs a new producer receipt or a replay with the \
                 original path"
            ),
            JoinError::Unverified { detail } => {
                write!(
                    f,
                    "the keys cannot be verified: {detail}; nothing was joined"
                )
            }
            JoinError::UnsupportedProducer { detail } => f.write_str(detail),
            JoinError::KeyKind { key, scope } => write!(
                f,
                "the key {key} is of a kind a scope over a {scope} does not have; nothing \
                 more was joined"
            ),
            JoinError::Unknown { key } => write!(
                f,
                "the key {key} names no original of the scope; nothing more was joined"
            ),
            JoinError::Ambiguous { key, original } => write!(
                f,
                "the key {key} names the original {}, which an earlier key named; nothing \
                 more was joined",
                original.0
            ),
            JoinError::Drift { detail } => f.write_str(detail),
            JoinError::Failed(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for JoinError {}

impl From<L2dError> for JoinError {
    fn from(error: L2dError) -> Self {
        JoinError::Failed(error)
    }
}

/// An incompatible mapping is a refusal: nothing was joined, and the message says why.
impl From<JoinError> for L2dError {
    fn from(error: JoinError) -> Self {
        match error {
            JoinError::Failed(error) => error,
            other => L2dError::Refused {
                reason: other.to_string(),
            },
        }
    }
}

/// A key as the join looks it up.
enum Key<'a> {
    Position(u64),
    StoreId(&'a str),
    PolicyId(&'a str),
}

fn kind_of(input: &MarkedInput) -> &'static str {
    match input.archive_entries {
        Some(_) => SourceKind::Archive.as_str(),
        None => SourceKind::File.as_str(),
    }
}

/// Compares what an artifact records of its input with what the scope binds: the
/// bytes, the entry filter and the reader interpretation, in that order.
fn require_input(artifact: &MarkedInput, scope: &MarkedInput) -> Result<(), JoinError> {
    if artifact.content_sha256 != scope.content_sha256 {
        return Err(JoinError::SourceChanged {
            artifact_sha256: artifact.content_sha256.clone(),
            scope_sha256: scope.content_sha256.clone(),
        });
    }
    match (&artifact.archive_entries, &scope.archive_entries) {
        (None, None) => {}
        (Some(made), Some(bound)) if made == bound => {}
        (Some(made), Some(bound)) => {
            return Err(JoinError::Filters {
                artifact: made.clone(),
                scope: bound.clone(),
            });
        }
        _ => {
            return Err(JoinError::SourceKind {
                artifact: kind_of(artifact),
                scope: kind_of(scope),
            });
        }
    }
    match (&artifact.interpretation, &scope.interpretation) {
        (None, _) => Err(JoinError::Unverified {
            detail: "the artifact does not record how its input was read".into(),
        }),
        (Some(made), Some(bound)) if made == bound => Ok(()),
        (Some(made), bound) => Err(JoinError::Interpretation {
            detail: format!("{made:?} against {bound:?}"),
        }),
    }
}

impl Scope {
    /// Compares the source a selection, a mark set or a score manifest records with
    /// this scope.
    fn require_source(&self, artifact: &MarkedSource) -> Result<(), JoinError> {
        match (artifact, self.source()) {
            (MarkedSource::Input(made), MarkedSource::Input(bound)) => {
                require_input(made, bound)?;
                if made.path == bound.path || artifact.same_history(self.source()) {
                    return Ok(());
                }
                Err(JoinError::Path {
                    artifact: made.path.clone(),
                    scope: bound.path.clone(),
                })
            }
            (MarkedSource::Store(made), MarkedSource::Store(bound)) => {
                if made.content_sha256 != bound.content_sha256 {
                    return Err(JoinError::SourceChanged {
                        artifact_sha256: made.content_sha256.clone(),
                        scope_sha256: bound.content_sha256.clone(),
                    });
                }
                if made.name != bound.name {
                    return Err(JoinError::Path {
                        artifact: format!("the store {}", made.name),
                        scope: format!("the store {}", bound.name),
                    });
                }
                Ok(())
            }
            (MarkedSource::Input(made), MarkedSource::Store(_)) => Err(JoinError::SourceKind {
                artifact: kind_of(made),
                scope: self.kind().as_str(),
            }),
            (MarkedSource::Store(_), MarkedSource::Input(_)) => Err(JoinError::SourceKind {
                artifact: SourceKind::Store.as_str(),
                scope: self.kind().as_str(),
            }),
        }
    }

    /// Compares the query source of a policy report with this scope, and returns the
    /// path its IDs are derived from.
    fn require_policy_source<'r>(
        &self,
        query_source: Option<&'r MarkedInput>,
    ) -> Result<&'r str, JoinError> {
        let (MarkedSource::Input(bound), Some(id_source)) =
            (self.source(), self.binding().policy_id_source.as_deref())
        else {
            return Err(JoinError::UnsupportedProducer {
                detail: "the policy decides queries of a file or an archive and names them by \
                         source path and text; a scope over a store has no such IDs, and no \
                         verified mapping from them to stored logs exists; nothing was joined"
                    .into(),
            });
        };
        let made = query_source.ok_or_else(|| JoinError::Unverified {
            detail: "the policy report does not record its query source, so its IDs are local \
                     to the report"
                .into(),
        })?;
        require_input(made, bound)?;
        if made.path != id_source {
            return Err(JoinError::PolicyPath {
                report: made.path.clone(),
                scope: id_source.to_string(),
            });
        }
        Ok(&made.path)
    }

    /// Joins keys made on `produced_under` (the source a selection, a mark set or a
    /// score manifest records) to the originals of this scope.
    ///
    /// `each` gets the index of the key and the original it names, in the order of
    /// the keys. An error voids the join: what `each` was given until then is no
    /// result.
    pub fn join_keys<'k>(
        &self,
        produced_under: &MarkedSource,
        keys: impl IntoIterator<Item = NativeKey<'k>>,
        budget: &mut MemoryBudget,
        each: impl FnMut(u64, &Original) -> Result<(), L2dError>,
    ) -> Result<JoinReport, JoinError> {
        self.require_source(produced_under)?;
        let keys = keys.into_iter().map(|key| {
            Ok(match key {
                NativeKey::Position(position) => Key::Position(position),
                NativeKey::StoreId(id) => Key::StoreId(id),
            })
        });
        self.join(keys, budget, each)
    }

    /// Joins the members of a saved selection to the originals of this scope. `each`
    /// gets every member with the original it names, in the selection's order. Errors
    /// as [`Scope::join_keys`].
    pub fn join_selection(
        &self,
        selection: &SelectionSet,
        budget: &mut MemoryBudget,
        mut each: impl FnMut(&Member, &Original) -> Result<(), L2dError>,
    ) -> Result<JoinReport, JoinError> {
        self.require_source(selection.source())?;
        let members = selection.members();
        let keys = members.iter().map(|member| {
            Ok(match member {
                Member::Position(position) => Key::Position(*position),
                Member::StoreId(id) => Key::StoreId(id),
            })
        });
        self.join(keys, budget, |index, original| {
            each(&members[index as usize], original)
        })
    }

    /// Joins the decisions of a saved policy report to the originals of this scope.
    ///
    /// The report's query source must be this scope's source, read the same way and
    /// opened at the same path: the IDs are derived from that path and the original
    /// text. Every decision's ID is checked against its own text first. `each` gets
    /// every decision with the original it names, in the report's order. Errors as
    /// [`Scope::join_keys`].
    pub fn join_policy(
        &self,
        report: &PolicyReport,
        budget: &mut MemoryBudget,
        mut each: impl FnMut(&PolicyDecision, &Original) -> Result<(), L2dError>,
    ) -> Result<JoinReport, JoinError> {
        let path = self.require_policy_source(report.query_source.as_ref())?;
        let decisions = &report.decisions;
        let keys = decisions.iter().map(|decision| {
            let input = &decision.input;
            if l2d_store::log_id(path, &input.text) != input.id {
                return Err(JoinError::Drift {
                    detail: format!(
                        "the policy original {} differs from the ID of its saved source path \
                         and text; nothing more was joined",
                        input.id
                    ),
                });
            }
            Ok(Key::PolicyId(&input.id))
        });
        self.join(keys, budget, |index, original| {
            each(&decisions[index as usize], original)
        })
    }

    fn join<'k>(
        &self,
        keys: impl Iterator<Item = Result<Key<'k>, JoinError>>,
        budget: &mut MemoryBudget,
        mut each: impl FnMut(u64, &Original) -> Result<(), L2dError>,
    ) -> Result<JoinReport, JoinError> {
        let before = budget.held_bytes();
        budget.require_whole(JOIN_FIXED_BYTES, "joining keys to a triager scope")?;
        budget.hold(before.saturating_add(JOIN_FIXED_BYTES));
        let mut seen = match self.index.seen(budget) {
            Ok(seen) => seen,
            Err(error) => {
                budget.hold(before);
                return Err(error.into());
            }
        };
        budget.hold(
            before
                .saturating_add(JOIN_FIXED_BYTES)
                .saturating_add(seen.held_bytes()),
        );
        let mut report = JoinReport::default();
        let result = keys.into_iter().try_for_each(|key| {
            let key = key?;
            let row = match &key {
                Key::Position(position) => match self.kind() {
                    SourceKind::Store => None,
                    SourceKind::File | SourceKind::Archive => {
                        Some(self.index.find_position(*position)?)
                    }
                },
                Key::StoreId(id) => match self.kind() {
                    SourceKind::Store => Some(self.find_id(id)?),
                    SourceKind::File | SourceKind::Archive => None,
                },
                Key::PolicyId(id) => Some(self.find_id(id)?),
            };
            let named = || match &key {
                Key::Position(position) => format!("position {position}"),
                Key::StoreId(id) => format!("store ID {id}"),
                Key::PolicyId(id) => format!("policy ID {id}"),
            };
            let row = row
                .ok_or_else(|| JoinError::KeyKind {
                    key: named(),
                    scope: self.kind().as_str(),
                })?
                .ok_or_else(|| JoinError::Unknown { key: named() })?;
            let original = self.original_at(row)?;
            if seen.mark(row)? {
                return Err(JoinError::Ambiguous {
                    key: named(),
                    original: original.id,
                });
            }
            each(report.keys, &original)?;
            report.keys += 1;
            report.originals += 1;
            report.occurrences += original.occurrences;
            Ok(())
        });
        budget.hold(before);
        result.map(|()| report)
    }

    /// The row of the original with the native identity `id`; `None` also when `id`
    /// is not an identity at all.
    fn find_id(&self, id: &str) -> Result<Option<u64>, L2dError> {
        match NativeId::from_hex(id) {
            Some(id) => self.index.find_id(&id.0),
            None => Ok(None),
        }
    }
}
