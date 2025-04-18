If you build OpenROAD with Rocky Linux 9 (which also works with Catapult):
- follow the instructions in OpenROAD/docs/user/Build.md
- You will likely see an error while installing pandoc. It comes from line 455 in the current version of DependencyInstaller.sh: 
tar xvzf pandoc-${pandocVersion}-linux-${arch}.tar.gz --strip-components 1 -C /usr/local/
- You can instead run the last few commands outside of the script as follows:

eval wget https://github.com/jgm/pandoc/releases/download/${pandocVersion}/pandoc-${pandocVersion}-linux-${arch}.tar.gz
sudo tar xvzf pandoc-${pandocVersion}-linux-${arch}.tar.gz --strip-components 1 -C /usr/local/
rm -rf pandoc-${pandocVersion}-linux-${arch}.tar.gz

^ make note of the prepended sudo command.

For the build script, if you run into an error with CMake where it can't find certain libraries, add this command before the cmake command at the end of the script:

export CMAKE_PREFIX_PATH=<your_install_path>

Note that the install path will probably pop up in the error message, so just use that.