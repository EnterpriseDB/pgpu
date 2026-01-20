#!/bin/bash

set -e

#--------------------------------------
# Variables
#--------------------------------------

export GPU_CUDA_TOOLKIT_VER=${GPU_CUDA_TOOLKIT_VER:-"13-0"}
export GPU_CUDA_ARCHITECTURES=${GPU_CUDA_ARCHITECTURES:-"89;90;100;103;120;121" } # 89: L40/L4, 90: H100/H200/GH200
export CUVS_VER=${CUVS_VER:-"25.12.00"}
export PG_VERSION=${PG_VERSION:-"pg17"}
# The cargo-pgrx version installed must match the version available in Cargo.toml
export CARGO_PGRX_REPO=https://github.com/pgcentralfoundation/pgrx
export CARGO_PGRX_REF_ARG="--tag v0.16.1"

#--------------------------------------
# Functions
#--------------------------------------

function main() {

  COMMAND=$1
  shift # Shift arguments so that $1 becomes $2, $2 becomes $3, etc.

  export PATH="${PATH}:$HOME/.local/bin"
  export PATH="${PATH}:$HOME/.cargo/bin"
  export PGRX_HOME="$HOME/.pgrx"

  case "$COMMAND" in
  setup)
    install_deps
    # setup vuVS rust env
    install_rust
    install_cargo_pgrx
    setup_cargo_pgrx_settings
    install_pg_extensions
    ;;
  build)
    build_extension
    ;;
  docker-build)
    docker build -t pgpu-ubuntu -f docker/Dockerfile.ubuntu .
    ;;
  docker-start)
    GPU_SUPPORT=
    if [ "$DISABLE_GPU_SUPPORT" != "1" ]; then
      GPU_SUPPORT="--gpus all"
    fi
    docker run -it --rm \
      --name pgpu-ubuntu \
      $GPU_SUPPORT \
      --network host \
      pgpu-ubuntu:latest
    ;;
  *)
    echo "ERROR: Invalid command '$COMMAND'"
    exit 1
    ;;
  esac
}

#--------------------------------------

function install_deps() {
  echo "Installing deps..."

  sudo apt update
  sudo apt install -y wget curl git \
    `# cargo-pgrx dependencies` \
    build-essential libssl-dev pkg-config \
    `# cargo-pgrx init dependencies` \
    libicu-dev bison flex libreadline-dev zlib1g-dev \
    `# extension dependencies` \
    libclang-dev
}

#--------------------------------------

function install_rust() {
  echo "Installing rust..."

  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  . "$HOME/.cargo/env"
  which cargo
}

#--------------------------------------

function install_cargo_pgrx() {
  echo "Installing cargo-pgrx for PostgreSQL support..."

  cargo install --git ${CARGO_PGRX_REPO} ${CARGO_PGRX_REF_ARG} cargo-pgrx --locked --force
  cargo pgrx init --$PG_VERSION download
}

#--------------------------------------

function install_pg_extensions() {
  echo "Installing extra extensions..."

  load_cargo_pgrx_env

  pip install pgxnclient
  pgxn install vchord
  pgxn install vector
}

#--------------------------------------

function setup_cargo_pgrx_settings() {

  echo "Initializing PGRX settings for [$PG_VERSION]..."

  # Collecting required PG instance information
  cargo pgrx status -vvv $PG_VERSION
  local PG_DATA=$(cargo pgrx status -vvv $PG_VERSION 2>&1 | grep "\-D" | sed 's/.* "//g' | sed 's/"$//g' | head -n1)
  local PG_CONF_FILE_NAME="postgresql.conf"
  local PG_CONF_PATH="${PG_DATA}/$PG_CONF_FILE_NAME"

  if [ ! -f $PG_CONF_PATH ]; then
    echo "⚠ WARNING: the $PG_CONF_FILE_NAME file was not found using 'cargo status' command, searching on [$PGRX_HOME]"
    PG_CONF_PATH=$(find $PGRX_HOME -iname "$PG_CONF_FILE_NAME" | head -n1)
    if [ ! -f "$PG_CONF_PATH" ]; then
      echo "✗ Error: $PG_CONF_FILE_NAME not found at [$PGRX_HOME]. Cannot proceed."
      exit 1
    fi
  fi

  # Set the proper variables
  MARKER="# extension extra settings" # The unique marker to check for
  if grep -qF "$MARKER" "$PG_CONF_PATH"; then
    echo "✓ Marker found in [$PG_CONF_PATH]. Settings assumed to be present. Skipping addition."
  else
    NEW_SETTINGS=$(
      cat <<EOF
# ------------------------------------
# $MARKER
# These settings are essential for the extension to run correctly.
shared_preload_libraries = 'vchord'
# ------------------------------------
EOF
    )
    echo "$NEW_SETTINGS" >>"$PG_CONF_PATH"
  fi
}

#--------------------------------------

function build_extension() {
  echo "Building the extension..."

  load_cargo_pgrx_env

  cargo pgrx install
}

#--------------------------------------

function load_cargo_pgrx_env() {
  export PATH="${PATH}:$(cargo pgrx info path "$PG_VERSION")/bin"
}

#--------------------------------------
# Entry point
#--------------------------------------
main $@
