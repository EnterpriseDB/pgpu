# NVIDIA repos use tags like "ubuntu2404"; see https://developer.download.nvidia.com/compute/cuda/repos
# so we template such a tag based on the name/version we have
export DIST_VERSION="9"
export GPU_CUDA_TOOLKIT_VER="12-9"
export GPU_FAISS_VER="v1.13.0"
#export GPU_CUDA_ARCHITECTURES="89;90;100;103;120;121" # 89: L40/L4, 90: H100/H200/GH200
export GPU_CUDA_ARCHITECTURES="89"


# we need packages from the subscription repos to build the NVIDIA driver kernel module
sudo subscription-manager register
sudo subscription-manager config --rhsm.manage_repos=1
sudo subscription-manager repos --enable=rhel-9-for-x86_64-appstream-rpms
sudo subscription-manager repos --enable=rhel-9-for-x86_64-baseos-rpms
sudo subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms

sudo dnf group install -y "Development Tools"
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

# MKL / OMP dependencies. FAISS need these -> can remove for CUVS
tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
sudo mv /tmp/oneAPI.repo /etc/yum.repos.d
sudo dnf install -y intel-oneapi-mkl-devel
sudo cp /opt/intel/oneapi/compiler/latest/lib/libiomp5.so /usr/lib64/


# CUDA
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${DIST_VERSION}/x86_64/cuda-rhel${DIST_VERSION}.repo
sudo dnf install -y cuda-toolkit-${GPU_CUDA_TOOLKIT_VER}

# NVIDIA driver; needed to run PGPU
sudo dnf install -y nvidia-driver-cuda


sudo dnf install -y pkg-config cmake
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

# FAISS
sudo mkdir /faiss_build
sudo chmod 777 -R /faiss_build
mkdir -p ~/faiss-git
cd faiss-git
git clone --branch ${GPU_FAISS_VER} --depth 1 https://github.com/facebookresearch/faiss.git
cd faiss/

# need to reduce opt level to avx2 since gcc on el8 only supports this. No problem for us; don't use these features of faiss
cmake -B /faiss_build -DFAISS_ENABLE_C_API=ON \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${GPU_CUDA_ARCHITECTURES}" \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DMKL_LIBRARIES=/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
cmake --build /faiss_build --parallel "$(nproc --ignore 1)"  # limit parallelism otherwise github runners crash/lose connection


# add the faiss and cuda libs to ldconfig so the linker can find them later when building aidb
sudo cp /faiss_build/c_api/libfaiss_c.so /usr/lib
sudo cp /faiss_build/faiss/libfaiss.so /usr/lib
sudo ldconfig -v


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
