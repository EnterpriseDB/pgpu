use cuvs::distance_type::DistanceType;
use ndarray::{ArrayBase, Axis, Ix2, OwnedRepr};

// TODO: can we consolidate this with the one in aidb-vectorchord-indexing/src/vectorchord_index.rs?
pub fn distance_type_from_str(s: &str) -> Option<DistanceType> {
    match s.to_lowercase().as_str() {
        "l2" => Some(DistanceType::L2Expanded),
        "ip" | "innerproduct" => Some(DistanceType::InnerProduct),
        "cos" | "cosine" => Some(DistanceType::CosineExpanded),
        _ => None,
    }
}

pub(crate) fn assert_valid_distance_operator(input: &str) {
    match input {
        "ip" | "l2" | "cos" => (),
        _ => {
            panic!("Invalid distance_operator \"{input}\": expected one of \"ip\", \"l2\", \"cos\"")
        }
    }
}

fn normalize_vectors(vecs: &mut ArrayBase<OwnedRepr<f32>, Ix2>) {
    for mut row in vecs.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt();
        // 3. Normalize in-place if there is normalization to be done
        if norm > f32::EPSILON {
            row /= norm;
        }
    }
}