use cuvs::distance_type::DistanceType;
use ndarray::{ArrayBase, Axis, Ix2, OwnedRepr};
use pgrx::Spi;

// TODO: consolidate distance op handling
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

pub(crate) fn normalize_vectors(vecs: &mut ArrayBase<OwnedRepr<f32>, Ix2>) {
    for mut row in vecs.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt();
        // Normalize in-place if there is normalization to be done
        if norm > f32::EPSILON {
            row /= norm;
        }
    }
}

pub fn print_memory(v: &Vec<f32>, message: &str) {
    let (heap_size, used) = calculate_vec_size(v);
    pgrx::debug1!("Memory footprint ({message}). Allocated: {heap_size:.2} GB, used: {used:.2} GB");
}

fn calculate_vec_size<T>(v: &Vec<T>) -> (f64, f64) {
    let elem_size = std::mem::size_of::<T>();
    let capacity = v.capacity();
    let length = v.len();
    (
        (capacity * elem_size) as f64 / 1024.0 / 1024.0 / 1024.0,
        (length * elem_size) as f64 / 1024.0 / 1024.0 / 1024.0,
    )
}

/// Parse a fully qualified table identifier using Postgres' built-in `parse_ident`.
///
/// # Panics
/// Panics if the identifier does not contain exactly `schema.table`.
pub fn parse_table_identifier(ident: &str) -> (String, String) {
    let parts: Vec<String> =
        Spi::get_one_with_args("SELECT pg_catalog.parse_ident($1)", &[ident.into()])
            .expect("parse_ident() returned NULL")
            .unwrap();

    if parts.len() != 2 {
        panic!(
            "Invalid identifier '{ident}': must contain schema and table name (e.g \"public.table\"), got {parts:?}",
        );
    }

    (parts[0].clone(), parts[1].clone())
}
