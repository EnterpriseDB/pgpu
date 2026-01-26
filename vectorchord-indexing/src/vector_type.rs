/// Decode a pgvector vector directly into a destination buffer (zero-copy where possible).
/// see reference: https://github.com/pgvector/pgvector/blob/a126c02184326287f8024bfce4c43d4e2fa099aa/src/vector.h#L11
/// Layout: [dim: u16] [unused: u16] [x: f32...]
///
/// Returns the dimension of the vector.
pub(crate) fn decode_pgvector_vector_into(byte_slice: &[u8], dest: &mut Vec<f32>) -> u32 {
    if byte_slice.len() < 4 {
        pgrx::error!("Invalid vector data: payload too short");
    }

    // Split off the 4-byte header (dim & unused)
    let (header_bytes, float_bytes) = byte_slice.split_at(4);

    // first 2 bytes are the 16-bit dimension
    let dim = u16::from_ne_bytes(header_bytes[0..2].try_into().unwrap());
    let dim_usize = dim as usize;

    // sanity checking
    if float_bytes.len() != dim_usize * 4 {
        pgrx::error!(
            "Vector dimension mismatch: Header says {}, found {} bytes (expected {})",
            dim,
            float_bytes.len(),
            dim_usize * 4
        );
    }

    // Zero-copy: reinterpret bytes as f32 slice and copy directly to destination.
    // This is safe because:
    // - pgvector stores f32 in native endian (same as our platform)
    // - The 4-byte header ensures float_bytes starts at a 4-byte aligned offset
    // - We're only reading, not writing through the pointer
    unsafe {
        let floats_ptr = float_bytes.as_ptr() as *const f32;
        let floats_slice = std::slice::from_raw_parts(floats_ptr, dim_usize);
        dest.extend_from_slice(floats_slice);
    }

    dim.into()
}
