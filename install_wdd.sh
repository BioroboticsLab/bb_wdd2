#!/bin/bash

# Source conda setup to ensure conda commands are available
eval "$(conda shell.bash hook)"

# Detect the operating system
OS=$(uname)

# Check if the 'wdd' environment exists, if not create it, otherwise activate it
if conda info --envs | grep -q "wdd"; then
    echo "Activating existing 'wdd' environment..."
else
    echo "Creating and activating new 'wdd' environment..."
    conda create -n wdd python=3.10 -y
fi

conda activate wdd

# Ensure pip is installed and available in the PATH
conda install -n wdd pip -y
export PATH="$CONDA_PREFIX/bin:$PATH"

# Now use pip as needed
pip install --upgrade pip

if [[ "$OS" == "Linux" ]]; then
    echo "Detected Linux OS"

    # Install essential tools for Linux
    echo "Installing essential tools with 'sudo apt install [package list]'"
    sudo apt update
    sudo apt install -y qtcreator qtbase5-dev qt5-qmake cmake libasound2-dev build-essential libglib2.0-dev libgirepository1.0-dev libarchive-dev gir1.2-glib-2.0 meson ninja-build

    # Install CUDA (Linux only)
    conda install -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.4 -y

    # Download and install Spinnaker SDK for Ubuntu
    echo "Downloading and installing Spinnaker SDK..."
    wget -O spinnaker-4.0.0.116-amd64-pkg-22.04.tar.gz https://flir.netx.net/file/asset/59513/original/attachment
    mkdir spinnaker-4
    tar -xvf spinnaker-4.0.0.116-amd64-pkg-22.04.tar.gz -C spinnaker-4/
    cd spinnaker-4/spinnaker-4.0.0.116-amd64/
    ./install_spinnaker.sh
    sudo systemctl restart udev  # this will restart udev, in case the error " /etc/init.d/udev: not found" comes up
    ## Respond to prompts: y to all, add your username to 'flirimaging' group, and enter sudo password when prompted
    cd ../..

    # Download and install Spinnaker Python bindings
    echo "Downloading and installing Spinnaker Python bindings..."
    wget -O spinnaker_python-4.0.0.116-cp310-cp310-linux_x86_64.tar.gz https://flir.netx.net/file/asset/59511/original/attachment
    mkdir spinnaker_python-4
    tar -xvf spinnaker_python-4.0.0.116-cp310-cp310-linux_x86_64.tar.gz -C spinnaker_python-4/
    cd spinnaker_python-4/

    pip install spinnaker_python-4.0.0.116-cp310-cp310-linux_x86_64.whl
    cd ..    

elif [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS"

    # Check if Homebrew is installed, install if not
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Ensure Homebrew is in the PATH
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo "Homebrew is already installed."
    fi

    # Install essential tools for macOS
    brew update
    brew install qt5 cmake libarchive meson ninja

    # Additional macOS-specific installations
    brew install portaudio  # Required for some audio-related dependencies

    # macOS does not support CUDA, so we skip CUDA installation
    conda install pytorch torchvision torchaudio -c pytorch -y
    # these were needed additionally for opencv to work
    conda install -c conda-forge liblapack lapack blas -y

    echo "Install script for Mac does not automatically install Spinnaker SDK and FLIR camera support. OpenCV video processing only."    

else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Install pip if not already installed
conda install pip -y

# Install Python libraries and dependencies using conda
conda install numpy=1.26 scipy scikit-image click iso8601 imageio hyperopt astropy asciimatics pyserial numba scikit-learn pandas tqdm matplotlib seaborn pytz -c conda-forge -y

# Install additional Python libraries using conda
conda install -c conda-forge opencv -y
conda install jupyter jupyterlab -y

# Install pip packages
pip install simple_pyspin vidgear simpleaudio multiprocessing_generator pysound pygobject pre-commit black madgrad imgaug

# Install wdd packages using pip
pip install git+https://github.com/BioroboticsLab/bb_wdd2.git
pip install git+https://github.com/BioroboticsLab/bb_wdd_filter.git
pip install git+https://github.com/walachey/wdd_bridge.git


echo "Installation completed successfully!"