use abi_stable::std_types::RVec;
use libloading::{Library, Symbol};

#[cfg(target_os = "linux")]
const PLUGIN_NAME: &str = "libpgpu_gpu_lib.so";
#[cfg(target_os = "macos")]
const PLUGIN_NAME: &str = "libpgpu_gpu_lib.dylib";

const PLUGIN_ABI_VERSION_EXPECT: u32 = 1;

// The function type we expect to find in the plugin.
// must match the signature from pgpu-gpu-lib/src/clustering_gpu_impl.rs
type RunClusteringFn = unsafe extern "C" fn(
    vectors: RVec<f32>,
    vector_dims: u32,
    cluster_count: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    spherical_centroids: bool,
) -> RVec<f32>;

type AbiVersionFn = unsafe extern "C" fn() -> u32;

pub(crate) fn run_clustering_in_plugin(
    vectors: Vec<f32>,
    vector_dims: u32,
    cluster_count: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    spherical_centroids: bool,
) -> Vec<f32> {
    let lib = unsafe { Library::new(PLUGIN_NAME) }.expect("failed to load plugin");
    let abi_version_fn: Symbol<AbiVersionFn> = unsafe {
        lib.get(b"plugin_abi_version\0")
            .expect("failed to get function symbol from plugin")
    };
    let abi_version_plugin = unsafe { abi_version_fn() };
    assert_eq!(
        abi_version_plugin, PLUGIN_ABI_VERSION_EXPECT,
        "plugin ABI version mismatch. Expected {}, got {}",
        PLUGIN_ABI_VERSION_EXPECT, abi_version_plugin
    );

    let run_clustering_fn: Symbol<RunClusteringFn> = unsafe {
        lib.get(b"run_clustering\0")
            .expect("failed to get function symbol from plugin")
    };

    pgrx::info!("calling pgpu_gpu_lib plugin");
    let result_centroids = unsafe {
        run_clustering_fn(
            vectors.into(),
            vector_dims,
            cluster_count,
            kmeans_iterations,
            kmeans_nredo,
            spherical_centroids,
        )
    };
    result_centroids.into_vec()
}
