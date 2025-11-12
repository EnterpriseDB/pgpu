// ABI version of the plugin. Increment this when the plugin interface changes.
const PLUGIN_ABI_VERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn plugin_abi_version() -> u32 {
    PLUGIN_ABI_VERSION
}
