cd /media/F/projects/moveai/codes/libs/openpose
VIDEO_DIR=$1
OUT_DIR=$2
for VIDEO_PATH in "$VIDEO_DIR"/*.mp4; do
  FILE_NAME=$(basename -- "$VIDEO_PATH")
  FILE_NAME="${FILE_NAME%.*}"
  OUT_JS_DIR="$OUT_DIR/$FILE_NAME"
  mkdir -p OUT_JS_DIR
  echo "$OUT_JS_DIR"
  ./build/examples/openpose/openpose.bin --video "$VIDEO_PATH" --write_json "$OUT_JS_DIR"
done