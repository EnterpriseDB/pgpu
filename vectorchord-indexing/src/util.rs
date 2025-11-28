use cuvs::distance_type::DistanceType;

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