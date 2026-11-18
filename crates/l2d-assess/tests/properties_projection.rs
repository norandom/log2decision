use l2d_assess::{VectorSet, assess_properties, project_to_csv};
use l2d_core::Label;
use std::{fs, path::PathBuf};

fn data(rows: Vec<Vec<f32>>) -> VectorSet {
    VectorSet {
        vectors: rows,
        labels: None,
        groups: None,
        name: "fixture".into(),
    }
}
fn output(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("l2d-35-{}-{name}.csv", std::process::id()))
}
#[test]
fn known_cosines_norms_and_numeric_duplicates() {
    let set = data(vec![
        vec![1., 0.],
        vec![1., -0.],
        vec![-2., 0.],
        vec![0., 1.],
    ]);
    let r = assess_properties(&set, 42).unwrap();
    assert_eq!((r.dimension, r.sample_size, r.seed), (2, 4, 42));
    assert_eq!(r.norm_quantiles, [1., 1., 1., 1.25, 2.]);
    assert_eq!(r.cosine_quantiles, [-1., -0.75, 0., 0., 1.]);
    assert!((r.anisotropy + 1. / 6.).abs() < 1e-6);
    assert_eq!(r.duplicate_rate, 0.25);
}

#[test]
fn projection_keeps_labels_groups_and_local_neighbours() {
    // The separation is in the third dimension. Copying the first two fails.
    let mut set = data(
        (0..32)
            .map(|i| vec![(i % 8) as f32 * 0.02, 0., (i / 8) as f32 * 8.])
            .collect(),
    );
    set.groups = Some((0..32).map(|i| format!("site,\"{i}\"\nline")).collect());
    set.labels = Some(
        (0..32)
            .map(|i| [Label::Good, Label::Bad, Label::Unknown][i % 3])
            .collect(),
    );
    let path = output("geometry");
    project_to_csv(&set, &path, 42).unwrap();
    let first = fs::read(&path).unwrap();
    let mut reader = csv::Reader::from_reader(first.as_slice());
    assert_eq!(
        reader.headers().unwrap().iter().collect::<Vec<_>>(),
        ["x", "y", "label", "group"]
    );
    let mut xy = Vec::<[f64; 2]>::new();
    for (i, record) in reader.records().enumerate() {
        let row = record.unwrap();
        assert_eq!(&row[2], set.labels.as_ref().unwrap()[i].as_str());
        assert_eq!(&row[3], set.groups.as_ref().unwrap()[i]);
        let point = [
            row[0].parse::<f64>().unwrap(),
            row[1].parse::<f64>().unwrap(),
        ];
        assert!(point.iter().all(|x| x.is_finite()));
        xy.push(point);
    }
    assert_eq!(xy.len(), 32);
    let matched = (0..32)
        .filter(|&i| {
            let distance = |j: usize| (xy[i][0] - xy[j][0]).hypot(xy[i][1] - xy[j][1]);
            let nearest = (0..32)
                .filter(|&j| j != i)
                .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
                .unwrap();
            nearest / 8 == i / 8
        })
        .count();
    assert!(
        matched >= 28,
        "{matched}/32 retained a same-cluster nearest neighbour"
    );
    project_to_csv(&set, &path, 42).unwrap();
    assert_eq!(fs::read(&path).unwrap(), first);
    project_to_csv(&set, &path, 7).unwrap();
    assert_ne!(fs::read(&path).unwrap(), first);
    fs::remove_file(path).unwrap();
}
