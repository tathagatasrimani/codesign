### NOTE: This script is set up to work on the RSG linux machines.

cd openroad_interface/OpenROAD

set +e
sudo ./etc/DependencyInstaller.sh -base 
status=$?
set -e

if [ $status -ne 0 ]; then
    echo "DependencyInstaller failed. Attempting manual pandoc install..."

    arch=amd64
    pandocVersion=3.1.11.1
    eval wget https://github.com/jgm/pandoc/releases/download/${pandocVersion}/pandoc-${pandocVersion}-linux-${arch}.tar.gz
    sudo tar xvzf pandoc-${pandocVersion}-linux-${arch}.tar.gz --strip-components 1 -C /usr/local/
    rm -rf pandoc-${pandocVersion}-linux-${arch}.tar.gz

fi

./etc/DependencyInstaller.sh -common -local

echo "\n\n\nOpenROAD dependencies installed successfully.\n\n\n"
echo "Installing OpenROAD..."

export CMAKE_PREFIX_PATH="/usr/local:/rsghome/$(whoami)/.local"

./etc/Build.sh

mkdir results


