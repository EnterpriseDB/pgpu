# NVIDIA repos use tags like "ubuntu2404"; see https://developer.download.nvidia.com/compute/cuda/repos
# so we template such a tag based on the name/version we have
export DIST_VERSION="9"
export GPU_CUDA_TOOLKIT_VER="12-9"
#export GPU_CUDA_ARCHITECTURES="89;90;100;103;120;121" # 89: L40/L4, 90: H100/H200/GH200
export GPU_CUDA_ARCHITECTURES="89"
export CUVS_VER="25.10.00"

# we need packages from the subscription repos to build the NVIDIA driver kernel module
sudo subscription-manager register
sudo subscription-manager config --rhsm.manage_repos=1
sudo subscription-manager repos --enable=rhel-9-for-x86_64-appstream-rpms
sudo subscription-manager repos --enable=rhel-9-for-x86_64-baseos-rpms
sudo subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms

sudo dnf group install -y "Development Tools"
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm


# CUDA
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${DIST_VERSION}/x86_64/cuda-rhel${DIST_VERSION}.repo
sudo dnf install -y cuda-toolkit-${GPU_CUDA_TOOLKIT_VER}

# NVIDIA driver; needed to run PGPU
sudo dnf install -y nvidia-driver-cuda


sudo dnf install -y pkg-config cmake
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc



sudo mkdir /miniconda
sudo chmod 777 -R /miniconda
curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -u -p /miniconda
source "/miniconda/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r



curl -fsSLo rust_cuda-129_arch-x86_64.yaml https://raw.githubusercontent.com/rapidsai/cuvs/refs/tags/v${CUVS_VER}/conda/environments/rust_cuda-129_arch-x86_64.yaml
conda env create -f rust_cuda-129_arch-x86_64.yaml
conda activate rust_cuda-129_arch-x86_64



# set up git authentication for the builder; some rust dependencies need to access private repos later
git config --global url."https://x:${GITHUB_TOKEN}@github.com".insteadOf "https://github.com"

# set up cargo/pgrx for extension build
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
cargo install --git https://github.com/EnterpriseDB/pgrx --branch develop-v0.16.1-edb cargo-pgrx --locked --force


# set up PG for extension development and testing
# Install the repository RPM:
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-x86_64/pgdg-redhat-repo-latest.noarch.rpm
# Disable the built-in PostgreSQL module:
sudo dnf -qy module disable postgresql

sudo dnf install -y postgresql18-server postgresql18-devel pgvector_18

cargo pgrx init --pg18 /usr/pgsql-18/bin/pg_config
sudo chown -R ec2-user:ec2-user /usr/pgsql-18/

# we also need "vectorchord"; install from pgxn
sudo dnf install -y pip
sudo pip install pgxnclient
export PATH=/usr/pgsql-18/bin:$PATH
pgxn install vchord

# no build/run PGPU
# cargo pgrx run pg18
# ALTER SYSTEM SET shared_preload_libraries = "vchord";
# then restart PG
