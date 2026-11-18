use crate::overlap::{SAMPLE_CAP, sample_indices};
use crate::{VectorSet, invalid};
use bhtsne::{Neighbor, tSNE};
use l2d_core::{L2dError, MemoryBudget, MemorySettings};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::path::Path;

/// Project one seeded sample to x,y,label,group in original row order.
/// Publish atomically after fitting and flushing every sampled row.
pub fn project_to_csv(set: &VectorSet, out: &Path, seed: u64) -> Result<(), L2dError> {
    project_to_csv_with(
        set,
        out,
        seed,
        &MemoryBudget::observe(&MemorySettings::default())?,
    )
}

pub fn project_to_csv_with(
    set: &VectorSet,
    out: &Path,
    seed: u64,
    budget: &MemoryBudget,
) -> Result<(), L2dError> {
    set.validate()?;
    let n = set.vectors.len().min(SAMPLE_CAP);
    if n < 4 {
        return Err(invalid(
            "t-SNE needs at least four sampled vectors for perplexity >= 1",
        ));
    }
    let perplexity = ((n - 1) / 3).min(20);
    let k = 3 * perplexity;
    // Precomputed neighbour graph, backend's symmetrized sparse affinities,
    // optimizer, adaptive quadtree, single worker stack, and CSV buffer.
    let work = n as u64 * (k as u64 * 96 + 65536) + (4 << 20);
    budget.require_whole(
        set.resident_bytes().saturating_add(work),
        "t-SNE projection inputs and workspace",
    )?;
    let mut sample = sample_indices(set.vectors.len(), seed);
    sample.sort_unstable();
    let rows: Vec<_> = sample.iter().map(|&i| set.vectors[i].as_slice()).collect();
    let graph = neighbours(&rows, k);
    let mut rng = SmallRng::seed_from_u64(seed);
    let initial: Vec<f64> = (0..2 * n).map(|_| rng.random_range(-1e-4..1e-4)).collect();
    // A single worker fixes floating-point reduction order regardless of
    // host core count or concurrently scheduled Rayon jobs.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(2 << 20)
        .build()
        .map_err(|e| L2dError::Runtime {
            detail: format!("start t-SNE worker: {e}"),
        })?;
    let xy = pool.install(|| {
        tSNE::new(&rows)
            .perplexity(perplexity as f64)
            .learning_rate((n as f64 / 12.).max(10.))
            .initial_embedding(initial)
            .epochs(500)
            .barnes_hut_with_neighbors(0.5, &graph)
            .embedding()
    });
    let (points, remainder) = xy.as_chunks::<2>();
    if points.len() != n || !remainder.is_empty() {
        return Err(L2dError::Runtime {
            detail: format!("t-SNE returned {} coordinates for {n} rows", xy.len()),
        });
    }
    if !xy.iter().all(|v| v.is_finite()) {
        return Err(invalid("t-SNE produced nonfinite coordinates"));
    }
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let failed = |e: std::io::Error| L2dError::Runtime {
        detail: format!("write projection {}: {e}", out.display()),
    };
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(failed)?;
    {
        let mut writer = csv::Writer::from_writer(file.as_file_mut());
        writer
            .write_record(["x", "y", "label", "group"])
            .map_err(std::io::Error::from)
            .map_err(failed)?;
        for (&i, point) in sample.iter().zip(points) {
            let label = set.labels.as_ref().map_or("", |v| v[i].as_str());
            let group = set.groups.as_ref().map_or("", |v| v[i].as_str());
            writer
                .write_record([&point[0].to_string(), &point[1].to_string(), label, group])
                .map_err(std::io::Error::from)
                .map_err(failed)?;
        }
        writer.flush().map_err(failed)?;
    }
    file.persist(out).map_err(|e| failed(e.error))?;
    Ok(())
}

// The backend's vantage-point tree randomizes pivots. Resolve equidistant
// neighbours by sample index before handing the sparse graph to Barnes-Hut.
fn neighbours(rows: &[&[f32]], k: usize) -> Vec<Vec<Neighbor<f64>>> {
    let mut scratch = Vec::with_capacity(rows.len() - 1);
    let mut graph = Vec::with_capacity(rows.len());
    for (i, &row) in rows.iter().enumerate() {
        scratch.clear();
        scratch.extend(
            rows.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(index, &other)| {
                    let distance = row
                        .iter()
                        .zip(other)
                        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    Neighbor { index, distance }
                }),
        );
        let order = |a: &Neighbor<f64>, b: &Neighbor<f64>| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.index.cmp(&b.index))
        };
        scratch.select_nth_unstable_by(k - 1, order);
        scratch[..k].sort_unstable_by(order);
        // Each Gaussian has its own fitted bandwidth. Scaling and shifting
        // squared distances preserves normalized affinities and prevents large
        // coordinates from underflowing even the closest neighbour to zero.
        let scale = scratch[k - 1].distance;
        if scale > 0. {
            let minimum = (scratch[0].distance / scale).powi(2);
            for neighbour in &mut scratch[..k] {
                neighbour.distance = ((neighbour.distance / scale).powi(2) - minimum).sqrt();
            }
        }
        graph.push(scratch[..k].to_vec());
    }
    graph
}
