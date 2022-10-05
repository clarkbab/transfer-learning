#! /usr/bin/env bash
live='https://upload.pypi.org/legacy/'
test='https://test.pypi.org/legacy/'

REQUIRED_PACKAGES=('setuptools' 'wheel' 'twine')

# Set server to argument or default.
server_name=${1-test}

case $server_name in
  'test')
    server=$test
    ;;
  'live')
    server=$live
    ;;
  *)
    echo "Unknown server ${server_name}."
    exit 1
    ;;
esac

# Install required Python packages
for package in ${REQUIRED_PACKAGES[@]}
do
  echo "Checking for Python package '${package}'..."

  # Check if package is already installed.
  pip show $package

  if [ $? -eq 1 ]
  then
    echo "Installing package $package..."
    pip install $package

    if [ $? -eq 1 ]
    then
      echo "Failed installing package $package."
    fi
    echo "Installed."
  else
    echo 'Already installed.'
  fi
done

# Clean dist folder.
echo "Cleaning dist folder..."
rm -rf dist

# Build distribution archives.
echo "Building distribution archives..."
python setup.py sdist bdist_wheel
echo "Built."

# Find twine location.
cmd="pip show twine | grep Location | sed 's/Location: //'"
location=$(eval $cmd)

# Upload the archives.
echo "Uploading archives to ${server_name} server..."
python ${location}/twine upload --repository-url ${server} dist/*