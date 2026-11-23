//! Membership of the originals of one snapshot in a named set: known-in, known-out or
//! not evaluated, and the three-valued expressions over such sets (design.md,
//! "Membership, cost and evidence").

use serde_json::{Value, json};

use super::wire::{Fault, MAX_SETS, share};

/// The membership of one original in one set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Out = 0,
    In = 1,
    /// Not evaluated: outside the producer's scope, or without a usable output.
    Unknown = 2,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Out => "out",
            State::In => "in",
            State::Unknown => "not_evaluated",
        }
    }

    fn of(bits: u8) -> Self {
        match bits & 3 {
            0 => State::Out,
            1 => State::In,
            _ => State::Unknown,
        }
    }

    pub fn and(self, other: State) -> State {
        match (self, other) {
            (State::Out, _) | (_, State::Out) => State::Out,
            (State::In, State::In) => State::In,
            _ => State::Unknown,
        }
    }

    pub fn or(self, other: State) -> State {
        match (self, other) {
            (State::In, _) | (_, State::In) => State::In,
            (State::Out, State::Out) => State::Out,
            _ => State::Unknown,
        }
    }

    /// The complement within the declared parent; what is unknown stays unknown.
    pub fn not(self) -> State {
        match self {
            State::In => State::Out,
            State::Out => State::In,
            State::Unknown => State::Unknown,
        }
    }
}

/// Two bits per row of a snapshot's relation.
pub(crate) struct States {
    bits: Vec<u8>,
    rows: u64,
}

impl States {
    /// What the states of `rows` rows take in memory.
    pub fn bytes_for(rows: u64) -> u64 {
        rows.div_ceil(4).max(1)
    }

    /// Every row in `fill`. The caller has asked the budget for [`States::bytes_for`].
    pub fn new(rows: u64, fill: State) -> Self {
        let pattern = (fill as u8) * 0b0101_0101;
        Self {
            bits: vec![pattern; Self::bytes_for(rows) as usize],
            rows,
        }
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn held_bytes(&self) -> u64 {
        self.bits.capacity() as u64
    }

    pub fn get(&self, row: u64) -> State {
        debug_assert!(row < self.rows);
        State::of(self.bits[(row / 4) as usize] >> ((row % 4) * 2))
    }

    pub fn set(&mut self, row: u64, state: State) {
        debug_assert!(row < self.rows);
        let shift = (row % 4) * 2;
        let byte = &mut self.bits[(row / 4) as usize];
        *byte = (*byte & !(3 << shift)) | ((state as u8) << shift);
    }
}

/// A set expression over named operands, by their index in the request's `sets`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Expr {
    Name(usize),
    Union(Vec<Expr>),
    Intersection(Vec<Expr>),
    Difference(Box<Expr>, Box<Expr>),
    /// What the parent holds and the operand does not; unknown stays unknown.
    Complement(Box<Expr>),
}

/// The most nodes of an expression, and how deep it may nest.
const MAX_NODES: usize = 64;
const MAX_DEPTH: usize = 8;

impl Expr {
    /// Reads the expression tree of a request: a set name, or an object with exactly
    /// one of `union`, `intersection` (two or more operands), `difference` (two) and
    /// `complement` (one, relative to the parent of the view).
    pub fn parse(value: &Value, names: &[&str]) -> Result<Self, Fault> {
        let mut nodes = 0;
        Self::parse_at(value, names, 0, &mut nodes)
    }

    fn parse_at(
        value: &Value,
        names: &[&str],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<Self, Fault> {
        *nodes += 1;
        if *nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(Fault::invalid(format!(
                "`expr` has more than {MAX_NODES} nodes or nests deeper than {MAX_DEPTH}"
            )));
        }
        match value {
            Value::String(name) => names
                .iter()
                .position(|known| known == name)
                .map(Expr::Name)
                .ok_or_else(|| {
                    Fault::invalid(format!(
                        "`expr` names the set `{}`, which `sets` does not define",
                        super::wire::clip(name, 64).0
                    ))
                }),
            Value::Object(fields) if fields.len() == 1 => {
                let (operator, operands) = fields.iter().next().expect("one field");
                if operator == "state" {
                    return Err(Fault::unsupported(
                        "the `state` selector of an expression selects by persisted \
                         membership state for the pools of an allocation; it comes with \
                         `mode=allocate` (task 2.2). Use `union`, `intersection` and \
                         `difference` over named sets",
                    ));
                }
                let operands = operands.as_array().ok_or_else(|| {
                    Fault::invalid(format!("`{operator}` in `expr` takes a list of operands"))
                })?;
                let parsed = |nodes: &mut usize| {
                    operands
                        .iter()
                        .map(|operand| Self::parse_at(operand, names, depth + 1, nodes))
                        .collect::<Result<Vec<_>, _>>()
                };
                match operator.as_str() {
                    "union" | "intersection" if operands.len() < 2 => Err(Fault::invalid(format!(
                        "`{operator}` in `expr` takes two or more operands"
                    ))),
                    "union" => Ok(Expr::Union(parsed(nodes)?)),
                    "intersection" => Ok(Expr::Intersection(parsed(nodes)?)),
                    "difference" if operands.len() != 2 => Err(Fault::invalid(
                        "`difference` in `expr` takes exactly two operands",
                    )),
                    "complement" if operands.len() != 1 => Err(Fault::invalid(
                        "`complement` in `expr` takes exactly one operand",
                    )),
                    "complement" => {
                        let mut one = parsed(nodes)?;
                        Ok(Expr::Complement(Box::new(one.pop().expect("one operand"))))
                    }
                    "difference" => {
                        let mut both = parsed(nodes)?;
                        let right = both.pop().expect("two operands");
                        let left = both.pop().expect("two operands");
                        Ok(Expr::Difference(Box::new(left), Box::new(right)))
                    }
                    other => Err(Fault::invalid(format!(
                        "`expr` uses `{}`; the operators are `union`, `intersection`, \
                         `difference` and `complement`",
                        super::wire::clip(other, 64).0
                    ))),
                }
            }
            _ => Err(Fault::invalid(
                "`expr` is a set name or an object with one of `union`, `intersection`, \
                 `difference` and `complement`",
            )),
        }
    }

    /// The membership of one original, given its membership in every operand.
    pub fn eval(&self, operand: &impl Fn(usize) -> State) -> State {
        match self {
            Expr::Name(index) => operand(*index),
            Expr::Union(parts) => parts
                .iter()
                .fold(State::Out, |state, part| state.or(part.eval(operand))),
            Expr::Intersection(parts) => parts
                .iter()
                .fold(State::In, |state, part| state.and(part.eval(operand))),
            Expr::Difference(left, right) => left.eval(operand).and(right.eval(operand).not()),
            Expr::Complement(inner) => inner.eval(operand).not(),
        }
    }

    /// The expression with every name replaced by `named`: the canonical form a view
    /// records and its handle is derived from.
    pub fn describe(&self, named: &impl Fn(usize) -> Value) -> Value {
        let list = |parts: &[Expr]| -> Vec<Value> {
            parts.iter().map(|part| part.describe(named)).collect()
        };
        match self {
            Expr::Name(index) => named(*index),
            Expr::Union(parts) => json!({"union": list(parts)}),
            Expr::Intersection(parts) => json!({"intersection": list(parts)}),
            Expr::Difference(left, right) => {
                json!({"difference": [left.describe(named), right.describe(named)]})
            }
            Expr::Complement(inner) => json!({"complement": [inner.describe(named)]}),
        }
    }
}

/// Checks the names of a request's `sets`: at most `most`, each a short identifier.
pub(crate) fn check_names<'a>(
    names: impl Iterator<Item = &'a String>,
    most: usize,
) -> Result<(), Fault> {
    let mut count = 0;
    for name in names {
        count += 1;
        let usable = (1..=16).contains(&name.len())
            && name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            && name.as_bytes()[0].is_ascii_lowercase();
        if !usable {
            return Err(Fault::invalid(
                "a name in `sets` is 1 to 16 lower-case letters, digits or '_' and starts \
                 with a letter",
            ));
        }
    }
    if count > most.min(MAX_SETS) {
        return Err(Fault::invalid(format!(
            "`sets` names {count} sets; at most {} are taken here",
            most.min(MAX_SETS)
        )));
    }
    Ok(())
}

/// Originals and occurrences by state, within one parent population.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Tally {
    pub originals: [u64; 3],
    pub occurrences: [u64; 3],
}

impl Tally {
    pub fn add(&mut self, state: State, occurrences: u64) {
        self.originals[state as usize] += 1;
        self.occurrences[state as usize] += occurrences;
    }

    pub fn parent(&self) -> (u64, u64) {
        (self.originals.iter().sum(), self.occurrences.iter().sum())
    }

    pub fn of(&self, state: State) -> (u64, u64) {
        (
            self.originals[state as usize],
            self.occurrences[state as usize],
        )
    }

    /// The reduction receipt of one unit: parent, in, out, unknown and the retained
    /// share, whose denominator is the parent.
    fn unit(&self, unit: &str, counts: &[u64; 3]) -> Value {
        let parent: u64 = counts.iter().sum();
        json!({
            "unit": unit,
            "parent": parent,
            "in": counts[State::In as usize],
            "out": counts[State::Out as usize],
            "unknown": counts[State::Unknown as usize],
            "retained_share": share(counts[State::In as usize], parent),
        })
    }

    pub fn counts(&self) -> Value {
        self.unit("distinct_originals", &self.originals)
    }

    pub fn occurrence_counts(&self) -> Value {
        self.unit("occurrences", &self.occurrences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [State; 3] = [State::Out, State::In, State::Unknown];

    #[test]
    fn states_keep_two_bits_a_row() {
        let mut states = States::new(11, State::Unknown);
        assert_eq!(states.held_bytes(), 3);
        assert!((0..11).all(|row| states.get(row) == State::Unknown));
        states.set(0, State::In);
        states.set(5, State::Out);
        states.set(10, State::In);
        states.set(10, State::Out);
        let found: Vec<State> = (0..11).map(|row| states.get(row)).collect();
        assert_eq!(found[0], State::In);
        assert_eq!(found[5], State::Out);
        assert_eq!(found[10], State::Out);
        assert_eq!(
            found
                .iter()
                .filter(|state| **state == State::Unknown)
                .count(),
            8
        );
        assert_eq!(States::new(0, State::Out).held_bytes(), 1);
    }

    #[test]
    fn the_operators_are_three_valued() {
        use State::{In, Out, Unknown};
        // Union is in when either is in; intersection is out when either is out.
        assert_eq!(In.or(Unknown), In);
        assert_eq!(Out.or(Unknown), Unknown);
        assert_eq!(Out.and(Unknown), Out);
        assert_eq!(In.and(Unknown), Unknown);
        for a in ALL {
            for b in ALL {
                assert_eq!(a.or(b), b.or(a));
                assert_eq!(a.and(b), b.and(a));
                // De Morgan holds with unknowns preserved.
                assert_eq!(a.or(b).not(), a.not().and(b.not()));
            }
            assert_eq!(a.not().not(), a);
        }
        // `in - not_evaluated` stays unknown; `out - anything` is out.
        let names = ["a", "b"];
        let difference = Expr::parse(&json!({"difference": ["a", "b"]}), &names).unwrap();
        let of = |a: State, b: State| difference.eval(&|index| [a, b][index]);
        assert_eq!(of(In, Unknown), Unknown);
        assert_eq!(of(In, Out), In);
        assert_eq!(of(In, In), Out);
        assert_eq!(of(Out, Unknown), Out);
        assert_eq!(of(Unknown, In), Out);
        assert_eq!(of(Unknown, Out), Unknown);
        // The complement keeps what was not evaluated apart from what is out.
        let complement =
            Expr::parse(&json!({"complement": [{"union": ["a", "b"]}]}), &names).unwrap();
        let none = |a: State, b: State| complement.eval(&|index| [a, b][index]);
        assert_eq!(none(Out, Out), In);
        assert_eq!(none(In, Unknown), Out);
        assert_eq!(none(Out, Unknown), Unknown);
    }

    #[test]
    fn expressions_are_read_strictly_and_described_canonically() {
        let names = ["a", "b", "c"];
        let expr = Expr::parse(
            &json!({"intersection": [{"difference": ["a", "b"]}, {"union": ["b", "c", "a"]}]}),
            &names,
        )
        .unwrap();
        assert_eq!(
            expr.describe(&|index| json!(format!("set-{index}"))),
            json!({"intersection": [{"difference": ["set-0", "set-1"]},
                                    {"union": ["set-1", "set-2", "set-0"]}]})
        );
        for (bad, needle) in [
            (json!("d"), "`d`"),
            (json!({"union": ["a"]}), "two or more"),
            (json!({"difference": ["a", "b", "c"]}), "exactly two"),
            (json!({"xor": ["a", "b"]}), "xor"),
            (json!({"complement": ["a", "b"]}), "exactly one"),
            (
                json!({"union": ["a", "b"], "difference": ["a", "b"]}),
                "set name",
            ),
            (json!(["a"]), "set name"),
            (json!({"union": "a"}), "list"),
        ] {
            let fault = Expr::parse(&bad, &names).unwrap_err();
            assert_eq!(fault.code, "invalid_arguments");
            assert!(fault.message.contains(needle), "{bad}: {}", fault.message);
        }
        let mut deep = json!("a");
        for _ in 0..10 {
            deep = json!({"union": [deep, "b"]});
        }
        assert!(Expr::parse(&deep, &names).is_err());
        assert_eq!(
            Expr::parse(&json!({"state": {"set": "a", "is": "in"}}), &names)
                .unwrap_err()
                .code,
            "unsupported_operation"
        );
    }

    #[test]
    fn a_tally_reconciles_to_its_parent() {
        let mut tally = Tally::default();
        tally.add(State::In, 3);
        tally.add(State::In, 1);
        tally.add(State::Out, 5);
        tally.add(State::Unknown, 2);
        assert_eq!(tally.parent(), (4, 11));
        assert_eq!(
            tally.counts(),
            json!({"unit": "distinct_originals", "parent": 4, "in": 2, "out": 1,
                   "unknown": 1, "retained_share": 0.5})
        );
        assert_eq!(tally.occurrence_counts()["in"], 4);
        assert_eq!(Tally::default().counts()["retained_share"], Value::Null);
    }
}
