
# Navigate to the dir with the cloned model repo
# Run it like this: source ./setup_improve.sh

data_dir="csa_data"

# Use subshell to protect user shell from changes:
(
set -eu
# Get current dir and model dir
model_path=$PWD
echo "Model path: $model_path"
model_name=$(echo "$model_path" | awk -F '/' '{print $NF}')
echo "Model name: $model_name"

# Download data (if needed)
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    source download_csa.sh
else
    echo "CSA data folder already exists"
fi

# Env var IMPROVE_DATA_DIR
export IMPROVE_DATA_DIR="./$data_dir/"

# Clone IMPROVE lib (if needed)
pushd ../
improve_lib_path=$PWD/IMPROVE
improve_branch="v0.1.0"
if [[ -d $improve_lib_path ]]; then
    echo "IMPROVE repo exists in ${improve_lib_path}"
else
    git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
fi

pushd IMPROVE
branch_name="$(git branch --show-current)"
if [[ "$branch_name" == "$improve_branch" ]]; then
    echo "On the correct branch, ${improve_branch}"
else
    git checkout $improve_branch
fi

popd
popd
echo $improve_lib_path > $PWD/.setup-python-path.sh
)
if (( $? != 0 ))
then
  echo "setup failed!"
  return 1
fi

# Run these environment changes in the user shell:
export PYTHONPATH=$( cat $PWD/.setup-python-path.sh )
export IMPROVE_DATA_DIR="./$data_dir/"

echo
echo "IMPROVE_DATA_DIR: $IMPROVE_DATA_DIR"
echo "PYTHONPATH: $PYTHONPATH"
