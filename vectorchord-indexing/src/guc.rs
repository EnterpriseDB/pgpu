use crate::guc::GpuAcceleration::Enable;
use pgrx::guc::*;

#[derive(Copy, Clone, Debug, PartialEq, PostgresGucEnum)]
#[name = "gpu_acceleration"]
pub enum GpuAcceleration {
    Enable,
    Disable,
}
static GPU_ACCELERATION: GucSetting<GpuAcceleration> =
    GucSetting::<GpuAcceleration>::new(GpuAcceleration::Enable);

pub fn init_guc() {
    GucRegistry::define_enum_guc(
        c"pgpu.gpu_acceleration",
        c"GPU acceleration mode",
        c"Controls GPU acceleration: enable, disable",
        &GPU_ACCELERATION,
        GucContext::Userset,
        GucFlags::CUSTOM_PLACEHOLDER,
    );
}

pub fn get_gpu_acceleration() -> GpuAcceleration {
    GPU_ACCELERATION.get()
}

pub fn use_gpu_acceleration() -> bool {
    let g = get_gpu_acceleration();
    g == Enable
}
