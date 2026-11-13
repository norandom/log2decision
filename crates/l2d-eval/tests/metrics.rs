//! The metrics against cases calculated by hand, with ties and degenerate inputs.

use l2d_eval::metrics::{
    average_precision, budget_shares, log_loss_sum, recall_at_budget, roc_auc, spread,
};

fn close(found: f64, expected: f64) {
    assert!(
        (found - expected).abs() < 1e-12,
        "found {found}, expected {expected}"
    );
}

const T: bool = true;
const F: bool = false;

#[test]
fn roc_auc_and_average_precision_of_four_logs_without_ties() {
    // Pairs (bad, good): (0.35, 0.1) right, (0.35, 0.4) wrong, (0.8, 0.1) right,
    // (0.8, 0.4) right: 3 of 4.
    let ratios = [0.1, 0.4, 0.35, 0.8];
    let bad = [F, F, T, T];
    close(roc_auc(&ratios, &bad).unwrap(), 0.75);
    // From the top: 0.8 bad (precision 1, recall 1/2), 0.4 good, 0.35 bad (precision
    // 2/3, recall 1): 1/2 * 1 + 1/2 * 2/3.
    close(average_precision(&ratios, &bad).unwrap(), 0.5 + 1.0 / 3.0);
}

#[test]
fn a_perfect_and_an_inverted_ranking() {
    let ratios = [0.9, 0.8, 0.2, 0.1];
    close(roc_auc(&ratios, &[T, T, F, F]).unwrap(), 1.0);
    close(average_precision(&ratios, &[T, T, F, F]).unwrap(), 1.0);
    close(roc_auc(&ratios, &[F, F, T, T]).unwrap(), 0.0);
    // Bad logs at places 3 and 4: 1/2 * 1/3 + 1/2 * 2/4.
    close(
        average_precision(&ratios, &[F, F, T, T]).unwrap(),
        1.0 / 6.0 + 0.25,
    );
}

#[test]
fn tied_ratios_count_half_in_roc_auc_and_as_one_threshold_in_average_precision() {
    // Pairs: (0.9, 0.5) 1, (0.9, 0.1) 1, (0.5, 0.5) 1/2, (0.5, 0.1) 1: 3.5 of 4.
    let ratios = [0.9, 0.5, 0.5, 0.1];
    let bad = [T, T, F, F];
    close(roc_auc(&ratios, &bad).unwrap(), 0.875);
    // Thresholds: 0.9 (1 bad of 1, recall 1/2), 0.5 (2 bad of 3, recall 1), 0.1.
    close(
        average_precision(&ratios, &bad).unwrap(),
        0.5 + 0.5 * 2.0 / 3.0,
    );
    // The tie the other way round gives the same figures: no order inside a tie.
    let swapped = [T, F, T, F];
    let ratios = [0.9, 0.5, 0.5, 0.1];
    close(roc_auc(&ratios, &swapped).unwrap(), 0.875);
}

#[test]
fn all_ratios_tied_is_chance() {
    let ratios = [0.5_f32; 6];
    let bad = [T, F, T, F, F, F];
    close(roc_auc(&ratios, &bad).unwrap(), 0.5);
    // One threshold: precision 2/6 at recall 1.
    close(average_precision(&ratios, &bad).unwrap(), 2.0 / 6.0);
}

#[test]
fn one_label_only_has_no_roc_auc() {
    assert_eq!(roc_auc(&[0.1, 0.9], &[T, T]), None);
    assert_eq!(roc_auc(&[0.1, 0.9], &[F, F]), None);
    assert_eq!(roc_auc(&[], &[]), None);
    assert_eq!(roc_auc(&[0.3], &[T]), None);
    // Average precision needs a bad log; without a good log every place is right.
    assert_eq!(average_precision(&[0.1, 0.9], &[F, F]), None);
    assert_eq!(average_precision(&[], &[]), None);
    close(average_precision(&[0.1, 0.9], &[T, T]).unwrap(), 1.0);
}

#[test]
fn the_metrics_do_not_depend_on_the_order_of_the_logs() {
    let ratios: Vec<f32> = (0..200).map(|i| ((i * 37) % 23) as f32 / 23.0).collect();
    let bad: Vec<bool> = (0..200).map(|i| (i * 11) % 7 < 3).collect();
    let mut order: Vec<usize> = (0..200).collect();
    order.sort_by_key(|&i| (i * 7919 + 5) % 211);
    let ratios2: Vec<f32> = order.iter().map(|&i| ratios[i]).collect();
    let bad2: Vec<bool> = order.iter().map(|&i| bad[i]).collect();
    // Bit for bit, not within a tolerance: reports must repeat exactly.
    assert_eq!(
        roc_auc(&ratios, &bad).unwrap().to_bits(),
        roc_auc(&ratios2, &bad2).unwrap().to_bits()
    );
    assert_eq!(
        average_precision(&ratios, &bad).unwrap().to_bits(),
        average_precision(&ratios2, &bad2).unwrap().to_bits()
    );
    assert_eq!(
        recall_at_budget(&ratios, &bad, 40).unwrap().to_bits(),
        recall_at_budget(&ratios2, &bad2, 40).unwrap().to_bits()
    );
}

#[test]
fn roc_auc_equals_the_count_over_all_pairs() {
    let ratios: Vec<f32> = (0..150).map(|i| ((i * 37) % 23) as f32 / 23.0).collect();
    let bad: Vec<bool> = (0..150).map(|i| (i * 11) % 7 < 3).collect();
    let (mut wins, mut pairs) = (0.0_f64, 0.0_f64);
    for (i, &b) in bad.iter().enumerate() {
        for (j, &g) in bad.iter().enumerate() {
            if b && !g {
                pairs += 1.0;
                wins += match ratios[i].partial_cmp(&ratios[j]).unwrap() {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Less => 0.0,
                };
            }
        }
    }
    close(roc_auc(&ratios, &bad).unwrap(), wins / pairs);
}

#[test]
fn a_tie_at_the_budget_is_shared_between_the_tied_logs() {
    let ratios = [0.9, 0.8, 0.8, 0.8, 0.1];
    let shares = budget_shares(&ratios, 2);
    close(shares[0], 1.0);
    for share in &shares[1..4] {
        close(*share, 1.0 / 3.0);
    }
    close(shares[4], 0.0);
    close(shares.iter().sum::<f64>(), 2.0);

    assert_eq!(budget_shares(&ratios, 0), vec![0.0; 5]);
    assert_eq!(budget_shares(&ratios, 5), vec![1.0; 5]);
    assert_eq!(budget_shares(&ratios, 99), vec![1.0; 5]);
    assert_eq!(budget_shares(&[], 3), Vec::<f64>::new());
    // A budget that ends between two ratios cuts cleanly.
    assert_eq!(budget_shares(&ratios, 4), vec![1.0, 1.0, 1.0, 1.0, 0.0]);
    assert_eq!(budget_shares(&ratios, 1), vec![1.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn recall_at_a_budget_is_the_share_of_bad_logs_inside_it() {
    let ratios = [0.9, 0.8, 0.8, 0.8, 0.1];
    let bad = [T, T, F, F, T];
    // Inside a budget of 2: the first, and a third of the second: (1 + 1/3) of 3.
    close(
        recall_at_budget(&ratios, &bad, 2).unwrap(),
        (1.0 + 1.0 / 3.0) / 3.0,
    );
    close(recall_at_budget(&ratios, &bad, 4).unwrap(), 2.0 / 3.0);
    close(recall_at_budget(&ratios, &bad, 5).unwrap(), 1.0);
    close(recall_at_budget(&ratios, &bad, 0).unwrap(), 0.0);
    assert_eq!(recall_at_budget(&ratios, &[F; 5], 2), None);
    assert_eq!(recall_at_budget(&[], &[], 2), None);
}

#[test]
fn log_loss_is_summed_and_stays_finite_at_zero_and_one() {
    let ln2 = std::f64::consts::LN_2;
    close(log_loss_sum(&[0.5, 0.5, 0.5], &[T, F, T]), 3.0 * ln2);
    close(
        log_loss_sum(&[0.25, 0.25], &[T, F]),
        -(0.25_f64.ln()) - (0.75_f64.ln()),
    );
    assert_eq!(log_loss_sum(&[], &[]), 0.0);
    // A ratio of exactly 1 for a good log, or 0 for a bad one, is clipped to the
    // smallest step of a 32-bit float below 1, so the loss is large and finite.
    let worst = -f64::from(f32::EPSILON).ln();
    close(log_loss_sum(&[1.0], &[F]), worst);
    close(log_loss_sum(&[0.0], &[T]), worst);
    assert!(log_loss_sum(&[1.0, 0.0], &[T, F]) < 1e-6);
}

#[test]
fn the_spread_of_a_list_of_figures() {
    let found = spread(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(found.count, 3);
    close(found.mean, 2.0);
    close(found.min, 1.0);
    close(found.max, 3.0);
    close(found.sd, 1.0);
    let one = spread(&[0.25]).unwrap();
    assert_eq!(
        (one.count, one.mean, one.min, one.max, one.sd),
        (1, 0.25, 0.25, 0.25, 0.0)
    );
    assert_eq!(spread(&[]), None);
}

#[test]
#[should_panic(expected = "not a number")]
fn a_ratio_that_is_not_a_number_is_not_ranked() {
    let _ = roc_auc(&[0.5, f32::NAN], &[T, F]);
}
