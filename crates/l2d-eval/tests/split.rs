//! Grouped folds: no group key crosses a fold boundary, the folds are a function of the
//! group keys and their sizes alone, and each rule gives the folds calculated by hand.

use std::collections::{BTreeMap, BTreeSet};

use l2d_core::L2dError;
use l2d_eval::split::{Folds, Unit, units};

/// 40 groups `/g00` to `/g39`; group `i` holds `1 + (i * 7) % 11` rows.
fn rows() -> Vec<String> {
    let mut rows = Vec::new();
    for group in 0..40 {
        for _ in 0..(1 + (group * 7) % 11) {
            rows.push(format!("/g{group:02}"));
        }
    }
    rows
}

/// A fixed permutation of `rows` that is not the identity.
fn shuffled(rows: &[String]) -> Vec<String> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&index| (index * 7919 + 13) % 10007);
    order.into_iter().map(|index| rows[index].clone()).collect()
}

fn set(keys: &[&str]) -> BTreeSet<String> {
    keys.iter().map(|key| key.to_string()).collect()
}

fn sized(sizes: &[(&str, usize)]) -> Vec<String> {
    sizes
        .iter()
        .flat_map(|(key, logs)| std::iter::repeat_n(key.to_string(), *logs))
        .collect()
}

/// Every group is held out in exactly one fold, and so is every row.
fn assert_partition(folds: &Folds, rows: &[String]) {
    let groups: BTreeSet<&String> = rows.iter().collect();
    for group in &groups {
        let holding: Vec<usize> = (0..folds.len())
            .filter(|&fold| folds.held_out(fold).contains(*group))
            .collect();
        assert_eq!(holding.len(), 1, "group {group} is held out in {holding:?}");
        assert_eq!(folds.fold_of(group), Some(holding[0]));
    }
    let mut times = vec![0_u32; rows.len()];
    for fold in 0..folds.len() {
        let mask = folds.mask(fold, rows);
        assert_eq!(mask.len(), rows.len());
        for (row, held) in mask.iter().enumerate() {
            times[row] += u32::from(*held);
            // The mask is the membership of the row's group: a group is never cut.
            assert_eq!(*held, folds.held_out(fold).contains(&rows[row]));
        }
    }
    assert!(
        times.iter().all(|&count| count == 1),
        "every row is held out exactly once"
    );
}

#[test]
fn units_count_the_rows_of_each_group_in_byte_order() {
    let found = units(&sized(&[("/b", 2), ("/a", 3), ("/c", 1)]), &[]).unwrap();
    assert_eq!(
        found,
        vec![
            Unit {
                key: "/a".into(),
                groups: vec!["/a".into()],
                logs: 3
            },
            Unit {
                key: "/b".into(),
                groups: vec!["/b".into()],
                logs: 2
            },
            Unit {
                key: "/c".into(),
                groups: vec!["/c".into()],
                logs: 1
            },
        ]
    );
}

#[test]
fn groups_named_together_become_one_unit_and_absent_names_are_ignored() {
    let rows = sized(&[("/b", 2), ("/a", 3), ("/c", 1), ("/d", 4)]);
    let together = vec![vec![
        "/d".to_string(),
        "/b".to_string(),
        "/absent".to_string(),
    ]];
    let found = units(&rows, &together).unwrap();
    assert_eq!(
        found,
        vec![
            Unit {
                key: "/a".into(),
                groups: vec!["/a".into()],
                logs: 3
            },
            Unit {
                key: "/b".into(),
                groups: vec!["/b".into(), "/d".into()],
                logs: 6
            },
            Unit {
                key: "/c".into(),
                groups: vec!["/c".into()],
                logs: 1
            },
        ]
    );
}

#[test]
fn a_group_named_in_two_sets_is_refused() {
    let rows = sized(&[("/a", 1), ("/b", 1), ("/c", 1)]);
    let together = vec![
        vec!["/a".to_string(), "/b".to_string()],
        vec!["/b".to_string(), "/c".to_string()],
    ];
    assert!(matches!(
        units(&rows, &together),
        Err(L2dError::InvalidInput { .. })
    ));
}

#[test]
fn balanced_folds_are_the_ones_calculated_by_hand() {
    // Largest first, each to the fold with the fewest rows, the lowest fold on a tie:
    // a(5) -> 0, b(4) -> 1, c(3) -> 2, d(2) -> 2 (3 rows), e(1) -> 1 (4 rows),
    // f(1) -> 0 (all hold 5).
    let rows = sized(&[
        ("/e", 1),
        ("/c", 3),
        ("/a", 5),
        ("/f", 1),
        ("/b", 4),
        ("/d", 2),
    ]);
    let folds = Folds::balanced(&units(&rows, &[]).unwrap(), 3).unwrap();
    assert_eq!(folds.len(), 3);
    assert_eq!(folds.held_out(0), &set(&["/a", "/f"]));
    assert_eq!(folds.held_out(1), &set(&["/b", "/e"]));
    assert_eq!(folds.held_out(2), &set(&["/c", "/d"]));
    assert_partition(&folds, &rows);
}

#[test]
fn no_group_crosses_a_fold_boundary_and_every_row_is_held_out_once() {
    let rows = rows();
    let all = units(&rows, &[]).unwrap();
    let first: BTreeSet<String> = (0..8).map(|group| format!("/g{:02}", group * 5)).collect();
    for folds in [
        Folds::balanced(&all, 5).unwrap(),
        Folds::balanced(&all, 2).unwrap(),
        Folds::dealt(&all, 5, &first).unwrap(),
        Folds::dealt(&all, 4, &BTreeSet::new()).unwrap(),
    ] {
        assert_partition(&folds, &rows);
        for fold in 0..folds.len() {
            assert!(!folds.held_out(fold).is_empty(), "fold {fold} is empty");
        }
    }
}

#[test]
fn balanced_folds_hold_nearly_the_same_number_of_rows() {
    let rows = rows();
    let folds = Folds::balanced(&units(&rows, &[]).unwrap(), 5).unwrap();
    let sizes: Vec<usize> = (0..5)
        .map(|fold| folds.mask(fold, &rows).iter().filter(|held| **held).count())
        .collect();
    let (least, most) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
    assert_eq!(sizes.iter().sum::<usize>(), rows.len());
    // No fold differs from another by more than the largest group (11 rows).
    assert!(most - least <= 11, "fold sizes {sizes:?}");
}

#[test]
fn the_folds_do_not_depend_on_the_order_of_the_rows() {
    let rows = rows();
    let other = shuffled(&rows);
    assert_ne!(rows, other);
    let (a, b) = (units(&rows, &[]).unwrap(), units(&other, &[]).unwrap());
    assert_eq!(a, b);
    assert_eq!(a.len(), 40);
    let first = set(&["/g03", "/g17", "/g30"]);
    assert_eq!(
        Folds::balanced(&a, 5).unwrap(),
        Folds::balanced(&b, 5).unwrap()
    );
    assert_eq!(
        Folds::dealt(&a, 5, &first).unwrap(),
        Folds::dealt(&b, 5, &first).unwrap()
    );
}

#[test]
fn the_same_folds_come_out_on_repeat() {
    let rows = rows();
    let all = units(&rows, &[]).unwrap();
    let folds = Folds::balanced(&all, 5).unwrap();
    assert_eq!(folds, Folds::balanced(&all, 5).unwrap());
    assert_eq!(
        (0..5).map(|fold| folds.held_out(fold).len()).sum::<usize>(),
        40
    );
}

#[test]
fn dealt_folds_follow_the_rule_of_the_science_gate() {
    // The named groups from the largest to the smallest, ties in byte order, dealt in
    // turn: m1(10) -> 0, m2(8) -> 1, m3(8) -> 2, m4(1) -> 0. The others in byte order,
    // dealt in turn from fold 0 again: s1 -> 0, s2 -> 1, s3 -> 2, s4 -> 0.
    let rows = sized(&[
        ("/s3", 1),
        ("/m3", 8),
        ("/s1", 2),
        ("/m4", 1),
        ("/m1", 10),
        ("/s4", 1),
        ("/m2", 8),
        ("/s2", 30),
    ]);
    let first = set(&["/m1", "/m2", "/m3", "/m4"]);
    let folds = Folds::dealt(&units(&rows, &[]).unwrap(), 3, &first).unwrap();
    assert_eq!(folds.held_out(0), &set(&["/m1", "/m4", "/s1", "/s4"]));
    assert_eq!(folds.held_out(1), &set(&["/m2", "/s2"]));
    assert_eq!(folds.held_out(2), &set(&["/m3", "/s3"]));
    assert_partition(&folds, &rows);
}

#[test]
fn groups_named_together_are_held_out_together_by_every_rule() {
    let rows = rows();
    let together = vec![vec!["/g04".to_string(), "/g31".to_string()]];
    let all = units(&rows, &together).unwrap();
    let first = set(&["/g04", "/g10", "/g20"]);
    let same_side = |folds: &Folds| {
        assert_partition(folds, &rows);
        assert_eq!(folds.fold_of("/g04"), folds.fold_of("/g31"));
        for fold in 0..folds.len() {
            let held = folds.held_out(fold);
            assert_eq!(held.contains("/g04"), held.contains("/g31"));
        }
    };
    for count in 2..=6 {
        same_side(&Folds::balanced(&all, count).unwrap());
        same_side(&Folds::dealt(&all, count, &first).unwrap());
    }
    // Without the constraint the two are apart under at least one of these rules, so
    // the assertions above are not empty.
    let free = units(&rows, &[]).unwrap();
    let apart = (2..=6).any(|count| {
        let folds = Folds::balanced(&free, count).unwrap();
        folds.fold_of("/g04") != folds.fold_of("/g31")
    });
    assert!(apart);
}

#[test]
fn leave_one_out_holds_one_unit_in_each_fold_and_nothing_else() {
    let rows = sized(&[("/a", 3), ("/b", 2), ("/c", 4), ("/d", 1)]);
    let together = vec![vec!["/c".to_string(), "/d".to_string()]];
    let all = units(&rows, &together).unwrap();
    let folds = Folds::leave_one_out(&all, &["/c".to_string(), "/a".to_string()]).unwrap();
    assert_eq!(folds.len(), 2);
    assert_eq!(folds.held_out(0), &set(&["/c", "/d"]));
    assert_eq!(folds.held_out(1), &set(&["/a"]));
    assert_eq!(folds.fold_of("/b"), None);
    assert_eq!(
        folds.mask(0, &rows),
        [vec![false; 5], vec![true; 5]].concat()
    );
    assert!(matches!(
        Folds::leave_one_out(&all, &["/nowhere".to_string()]),
        Err(L2dError::InvalidInput { .. })
    ));
    assert!(matches!(
        Folds::leave_one_out(&all, &["/a".to_string(), "/a".to_string()]),
        Err(L2dError::InvalidInput { .. })
    ));
    assert!(matches!(
        Folds::leave_one_out(&all, &[]),
        Err(L2dError::InvalidInput { .. })
    ));
}

#[test]
fn folds_that_cannot_be_made_are_refused() {
    let all = units(&sized(&[("/a", 3), ("/b", 2)]), &[]).unwrap();
    for folds in [0, 1, 3] {
        assert!(
            matches!(
                Folds::balanced(&all, folds),
                Err(L2dError::InvalidInput { .. })
            ),
            "{folds} balanced folds of two groups"
        );
        assert!(
            matches!(
                Folds::dealt(&all, folds, &BTreeSet::new()),
                Err(L2dError::InvalidInput { .. })
            ),
            "{folds} dealt folds of two groups"
        );
    }
    assert!(Folds::balanced(&all, 2).is_ok());
}

#[test]
fn the_membership_hash_is_the_sha256_of_the_sorted_keys_joined_by_line_feeds() {
    // SHA-256 of "/a\n/b\n/c" and of "/only" (computed outside this code).
    let rows = sized(&[("/c", 1), ("/a", 1), ("/b", 1), ("/only", 9)]);
    let together = vec![vec!["/a".to_string(), "/b".to_string(), "/c".to_string()]];
    let folds = Folds::balanced(&units(&rows, &together).unwrap(), 2).unwrap();
    assert_eq!(folds.held_out(0), &set(&["/only"]));
    assert_eq!(
        folds.membership_sha256(0),
        "3593e7110948f98430df98371e4dd93f2bd166fd0333bc0e16a06425a2ce8470"
    );
    assert_eq!(
        folds.membership_sha256(1),
        "d4be2c83c9e5a76d9153c839342406a6c160b6a6ad1e81dc532231f12e70e72f"
    );
}

#[test]
fn the_rows_per_fold_add_up() {
    let rows = rows();
    let folds = Folds::balanced(&units(&rows, &[]).unwrap(), 5).unwrap();
    let mut per_fold: BTreeMap<usize, usize> = BTreeMap::new();
    for row in &rows {
        *per_fold.entry(folds.fold_of(row).unwrap()).or_default() += 1;
    }
    assert_eq!(per_fold.values().sum::<usize>(), rows.len());
    assert_eq!(per_fold.len(), 5);
}
