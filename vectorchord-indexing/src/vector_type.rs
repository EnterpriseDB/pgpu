/// decode a byte array representing a pgvector vector into a Vec<f32>
/// see reference: https://github.com/pgvector/pgvector/blob/a126c02184326287f8024bfce4c43d4e2fa099aa/src/vector.h#L11
//  Layout: [dim: u16] [unused: u16] [x: f32...]
pub(crate) fn decode_pgvector_vector(byte_slice: &[u8]) -> (Vec<f32>, u32) {
    if byte_slice.len() < 4 {
        pgrx::error!("Invalid vector data: payload too short");
    }

    // Split off the 4-byte header (dim & unused)
    let (header_bytes, float_bytes) = byte_slice.split_at(4);

    // first 2 bytes are the 16-bit dimension
    // we use "native endian" here; technically, I think it should "always" be little endian
    let dim = u16::from_ne_bytes(header_bytes[0..2].try_into().unwrap());

    // some sanity checking
    if float_bytes.len() % 4 != 0 {
        pgrx::error!("Invalid vector data: byte length not a multiple of 4. We expect a seried of 4-byte float32/float4 values.");
    }
    if (float_bytes.len() / 4) != dim as usize {
        pgrx::error!(
            "Vector dimension mismatch: Header says {}, found {} floats",
            dim,
            float_bytes.len() / 4
        );
    }

    // create the vec with newly allocated / rust-owned memory. The PG memory can now be freed safely.
    let vector_values: Vec<f32> = float_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    (vector_values, dim.into())
}
