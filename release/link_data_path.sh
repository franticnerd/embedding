
# if [ "$(uname)" == "Darwin" ]; then

case "$(uname -s)" in
  Darwin)
    echo 'Link data path for Mac OS'
    ln -s ~/data/projects/crossmap/ ./data
    ;;
  Linux)
    echo 'Link data path for Linux'
    ;;
esac

